// Copyright (c) 2016 Stefan Guthe / GCC / TU-Darmstadt. All rights reserved. 
// Use of this source code is governed by the BSD 3-Clause license that can be
// found in the LICENSE file.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "CudaIntrinsics.h"

namespace CompressedVolumePrivate
{
	__constant__ unsigned int swizzle_wavelet[32] = {
		0x00, 0x08, 0x01, 0x09,
		0x10, 0x18, 0x11, 0x19,
		0x02, 0x0a, 0x03, 0x0b,
		0x12, 0x1a, 0x13, 0x1b,

		0x04, 0x0c, 0x05, 0x0d,
		0x14, 0x1c, 0x15, 0x1d,
		0x06, 0x0e, 0x07, 0x0f,
		0x16, 0x1e, 0x17, 0x1f };

	__constant__ unsigned int swizzle_regular[32] = {
		0x00, 0x01, 0x08, 0x09,
		0x02, 0x03, 0x0a, 0x0b,
		0x10, 0x11, 0x18, 0x19,
		0x12, 0x13, 0x1a, 0x1b,

		0x04, 0x05, 0x0c, 0x0d,
		0x06, 0x07, 0x0e, 0x0f,
		0x14, 0x15, 0x1c, 0x1d,
		0x16, 0x17, 0x1e, 0x1f };

	template <typename AT>
	__device__ __forceinline__ unsigned char getComp(const unsigned char *comp, AT off)
	{
		return comp[off];
	}

	template <typename V, typename AT>
	__device__ __forceinline__ unsigned int getRaw(const unsigned char *comp, AT off);

	template <>
	__device__ __forceinline__ unsigned int getRaw<unsigned char, unsigned int>(const unsigned char *comp, unsigned int off) { return getComp(comp, off); }

	template <>
	__device__ __forceinline__ unsigned int getRaw<unsigned short, unsigned int>(const unsigned char *comp, unsigned int off) { return ((unsigned short)getComp(comp, off)) + (((unsigned short)getComp(comp, off + 1)) << 8); }

	template <>
	__device__ __forceinline__ unsigned int getRaw<unsigned char, uint64>(const unsigned char *comp, uint64 off) { return getComp(comp, off); }

	template <>
	__device__ __forceinline__ unsigned int getRaw<unsigned short, uint64>(const unsigned char *comp, uint64 off) { return ((unsigned short)getComp(comp, off)) + (((unsigned short)getComp(comp, off + 1)) << 8); }

	template <typename V>
	__device__ __forceinline__ unsigned int getInc();

	template<>
	__device__ __forceinline__ unsigned int getInc<unsigned char>() { return 1; }

	template<>
	__device__ __forceinline__ unsigned int getInc<unsigned short>() { return 2; }

	// decompress8split always starts on a byte boundary
	template <typename V, typename AT>
	__device__ __forceinline__ void decompress8split(int &out, const unsigned char *comp, AT &base_off, unsigned int bits);

	template <>
	__device__ __forceinline__ void decompress8split<unsigned char>(int &out, const unsigned char *comp, unsigned int &base_off, unsigned int bits)
	{
		unsigned int start = (__laneid() & 7) * bits;
		unsigned int off = start >> 3;
		start &= 0x7;
		unsigned int tmp = getComp(comp, base_off + off);
		tmp += __shfl_down(tmp, 1) << 8;
		out = __bfe(tmp, start, bits);
	}

	template <>
	__device__ __forceinline__ void decompress8split<unsigned short>(int &out, const unsigned char *comp, unsigned int &base_off, unsigned int bits)
	{
		unsigned int start = (__laneid() & 7) * bits;
		unsigned int off = start >> 3;
		start &= 0x7;
		unsigned int tmp = getComp(comp, base_off + off);
		unsigned int shift;
		if (bits <= 8)
			shift = 8;
		else
		{
			tmp += ((unsigned int)getComp(comp, base_off + off + 1)) << 8;
			shift = 16;
		}
		tmp += __shfl_down(tmp, 1) << shift;
		out = __bfe(tmp, start, bits);
	}

	template <>
	__device__ __forceinline__ void decompress8split<unsigned char>(int &out, const unsigned char *comp, uint64 &base_off, unsigned int bits)
	{
		unsigned int start = (__laneid() & 7) * bits;
		unsigned int off = start >> 3;
		start &= 0x7;
		unsigned int tmp = getComp(comp, base_off + off);
		tmp += __shfl_down(tmp, 1) << 8;
		out = __bfe(tmp, start, bits);
	}

	template <>
	__device__ __forceinline__ void decompress8split<unsigned short>(int &out, const unsigned char *comp, uint64 &base_off, unsigned int bits)
	{
		unsigned int start = (__laneid() & 7) * bits;
		unsigned int off = start >> 3;
		start &= 0x7;
		unsigned int tmp = getComp(comp, base_off + off);
		unsigned int shift;
		if (bits <= 8)
			shift = 8;
		else
		{
			tmp += ((unsigned int)getComp(comp, base_off + off + 1)) << 8;
			shift = 16;
		}
		tmp += __shfl_down(tmp, 1) << shift;
		out = __bfe(tmp, start, bits);
	}

	__constant__ int delta_mask[2] = { 0x00000000, 0xffffffff };

	__device__ __forceinline__ int decodeDelta(const int &delta, const int &pred, const int &min, const int &max)
	{
		const int max_pos = max - pred;
		const int max_neg = pred - min;
		// this is a corner case where the whole range is positive only
		if (max_neg == 0) return pred + delta;

		const int balanced_max = __min(max_neg - 1, max_pos);
		if ((delta >> 1) > balanced_max)
		{
			if (max_pos >= max_neg)
			{
				return pred + delta - balanced_max - 1;
			}
			else
			{
				return pred - delta + balanced_max;
			}
		}
		return pred + ((delta >> 1) ^ delta_mask[delta & 1]);
	}

	template <typename V, typename AT>
	__device__ __forceinline__ void decompressRBUC8x8_decode_device(const unsigned char *comp, AT &off, int header, int &tmp0, int &tmp1)
	{
		int decomp_bits, bits, tmp_0, tmp_1, tmp_2;
		if (__laneid() < 8) decompress8split<unsigned char>(decomp_bits, comp, off, header & 0x3f);
		off += header & 0x3f;
		tmp_0 = __shfl(decomp_bits, 0);
		tmp_1 = __shfl(decomp_bits, 1);
		tmp_2 = __shfl(decomp_bits, 2);
		bits = __shfl(decomp_bits, __laneid() >> 3);
		if (__laneid() >= 8) off += tmp_0;
		if (__laneid() >= 16) off += tmp_1;
		if (__laneid() >= 24) off += tmp_2;
		decompress8split<V>(tmp0, comp, off, bits);
		tmp_0 = __shfl(decomp_bits, (__laneid() >> 3) + 1);
		tmp_1 = __shfl(decomp_bits, (__laneid() >> 3) + 2);
		tmp_2 = __shfl(decomp_bits, (__laneid() >> 3) + 3);
		off += bits + tmp_0 + tmp_1 + tmp_2;
		bits = __shfl(decomp_bits, (__laneid() >> 3) + 4);
		decompress8split<V>(tmp1, comp, off, bits);
		off += bits;
		off = __shfl(off, 31);
	}

	__device__ __forceinline__ void decompressRBUC8x8_submin_device(int &tmp0, int &tmp1, const int &min, const int &max)
	{
		// get the data into the right order
		tmp0 = __shfl(tmp0, swizzle_regular[__laneid()]);
		tmp1 = __shfl(tmp1, swizzle_regular[__laneid()]);

		tmp0 += min;
		tmp1 += min;
	}

	__device__ __forceinline__ void decompressRBUC8x8_submax_device(int &tmp0, int &tmp1, const int &min, const int &max)
	{
		// get the data into the right order
		tmp0 = __shfl(tmp0, swizzle_regular[__laneid()]);
		tmp1 = __shfl(tmp1, swizzle_regular[__laneid()]);

		tmp0 = max - tmp0;
		tmp1 = max - tmp1;
	}

	__device__ __forceinline__ void decompressRBUC8x8_haar_device(int &tmp0, int &tmp1, const int &min, const int &max)
	{
		// the swizzle for wavelets is rather bad since we either need two tmp0 or two tmp1
		// fixing this during decompression is even worse as it adds another four shuffle instructions rather than just two
		int t0, t1, t2, t3;
		t0 = __shfl(tmp0, swizzle_wavelet[__laneid() & 0xf]);
		t1 = __shfl(tmp0, swizzle_wavelet[0x10 + (__laneid() & 0xf)]);
		t2 = __shfl(tmp1, swizzle_wavelet[__laneid() & 0xf]);
		t3 = __shfl(tmp1, swizzle_wavelet[0x10 + (__laneid() & 0xf)]);
		if (__laneid() < 16)
		{
			tmp0 = t0;
			tmp1 = t1;
		}
		else
		{
			tmp0 = t2;
			tmp1 = t3;
		}

		// recover sign
		tmp0 = ((tmp0 >> 1) ^ delta_mask[tmp0 & 1]);
		tmp1 = ((tmp1 >> 1) ^ delta_mask[tmp1 & 1]);

		// z, distance 2
		if ((__laneid() & 0x15) == 0)
		{
			tmp0 -= tmp1 >> 1;
			tmp1 += tmp0;
		}

		// y, distance 2
		t0 = __shfl_down(tmp0, 8);
		t1 = __shfl_down(tmp1, 8);
		if ((__laneid() & 0x1d) == 0)
		{
			tmp0 -= t0 >> 1;
			tmp1 -= t1 >> 1;
		}
		t0 = __shfl_up(tmp0, 8);
		t1 = __shfl_up(tmp1, 8);
		if ((__laneid() & 0x1d) == 0x08)
		{
			tmp0 += t0;
			tmp1 += t1;
		}

		// x, distance 2
		t0 = __shfl_down(tmp0, 2);
		t1 = __shfl_down(tmp1, 2);
		if ((__laneid() & 0x17) == 0)
		{
			tmp0 -= t0 >> 1;
			tmp1 -= t1 >> 1;
		}
		t0 = __shfl_up(tmp0, 2);
		t1 = __shfl_up(tmp1, 2);
		if ((__laneid() & 0x17) == 0x02)
		{
			tmp0 += t0;
			tmp1 += t1;
		}

		// z, distance 1
		t0 = __shfl_down(tmp0, 16);
		t1 = __shfl_down(tmp1, 16);
		if ((__laneid() & 0x10) == 0)
		{
			tmp0 -= t0 >> 1;
			tmp1 -= t1 >> 1;
		}
		t0 = __shfl_up(tmp0, 16);
		t1 = __shfl_up(tmp1, 16);
		if ((__laneid() & 0x10) == 0x10)
		{
			tmp0 += t0;
			tmp1 += t1;
		}

		// y, distance 1
		t0 = __shfl_down(tmp0, 4);
		t1 = __shfl_down(tmp1, 4);
		if ((__laneid() & 0x04) == 0)
		{
			tmp0 -= t0 >> 1;
			tmp1 -= t1 >> 1;
		}
		t0 = __shfl_up(tmp0, 4);
		t1 = __shfl_up(tmp1, 4);
		if ((__laneid() & 0x04) == 0x04)
		{
			tmp0 += t0;
			tmp1 += t1;
		}

		// x, distance 1
		t0 = __shfl_down(tmp0, 1);
		t1 = __shfl_down(tmp1, 1);
		if ((__laneid() & 0x01) == 0)
		{
			tmp0 -= t0 >> 1;
			tmp1 -= t1 >> 1;
		}
		t0 = __shfl_up(tmp0, 1);
		t1 = __shfl_up(tmp1, 1);
		if ((__laneid() & 0x01) == 0x01)
		{
			tmp0 += t0;
			tmp1 += t1;
		}

		const int avg = ((int)min + (int)max + 1) >> 1;
		tmp0 += avg;
		tmp1 += avg;
	}

	__device__ __forceinline__ void decompressRBUC8x8_gradient_device(int &tmp0, int &tmp1, const int &min, const int &max)
	{
		// get the data into the right order
		tmp0 = __shfl(tmp0, swizzle_regular[__laneid()]);
		tmp1 = __shfl(tmp1, swizzle_regular[__laneid()]);

		// there are 10 overlapping interations 8 iterations for tmp0 and the 8 for tmp1
		const bool d_x = ((__laneid() & 3) != 0);
		const bool d_y = (((__laneid() >> 2) & 3) != 0);
		const bool d_z = (__laneid() >= 16);
		const int iter = (__laneid() & 3) + ((__laneid() >> 2) & 3) + (__laneid() >> 4);
		// if used, last is the value that has last been reconstructed by this thread, making the whole shuffling easier
		// prior stores the value before that
		// first iteration, only one value possible
		if (__laneid() == 0)
		{
			// 0: (0, 0, 0)
			const int pred = ((int)min + (int)max + 1) >> 1;
			tmp0 = decodeDelta(tmp0, pred, min, max);
		}
		// second iteration, only one delta possible __laneid(): 1, 4, 16 valid: 0
		{
			// the only value we are going to use is the first one (thread = 0)
			//  1 : (0, 0, 0) = 0 -> (1, 0, 0)
			//  4 : (0, 0, 0) = 0 -> (0, 1, 0)
			// 16 : (0, 0, 0) = 0 -> (0, 0, 1)
			bool first = (iter == 1);
			int pred = __shfl(tmp0, 0);
			if (first)
				tmp0 = decodeDelta(tmp0, pred, min, max);
		}
		int last, prior;
		// third iteration __laneid(): 0, 2, 5, 8, 17, 20 last: 0, 1, 4, 16
		{
			// still better to broadcast all four values
			//  0*: -> (0, 0, 0)
			//  2 : (1, 0, 0) =  1                                  -> (2, 0, 0)
			//  8 : (0, 1, 0) =  4                                  -> (0, 2, 0)
			//  0 : (0, 0, 1) = 16                                  -> (0, 0, 2)
			//  5 : (1, 0, 0) =  1 + (0, 1, 0) =  4 - (0, 0, 0) = 0 -> (1, 1, 0)
			// 17 : (1, 0, 0) =  1 + (0, 0, 1) = 16 - (0, 0, 0) = 0 -> (1, 0, 1)
			// 20 : (0, 1, 0) =  4 + (0, 0, 1) = 16 - (0, 0, 0) = 0 -> (0, 1, 1)
			bool first = (iter == 2);
			bool second = (__laneid() == 0);
			int tmp_0, tmp_1, tmp_4, tmp_16;
			int pred = 0;
			tmp_0 = __shfl(tmp0, 0);
			tmp_1 = __shfl(tmp0, 1);
			tmp_4 = __shfl(tmp0, 4);
			tmp_16 = __shfl(tmp0, 16);

			if (d_x) pred += tmp_1;
			if (d_y) pred += tmp_4;
			if (d_z || second) pred += tmp_16;
			if ((d_x && (d_y || d_z || second)) || (d_y && (d_z || second))) pred -= tmp_0;

			// move to 0*
			if (second)
			{
				prior = tmp0;
				last = tmp1;
			}
			else
			{
				// initialize all last values
				last = tmp0;
			}
			if (first || second)
			{
				if (pred < min) pred = min;
				if (pred > max) pred = max;
				last = decodeDelta(last, pred, min, max);
			}
			if (first)
				tmp0 = last;
			else if (second)
				tmp1 = last;
		}
		// fourth iteration last: 0, 2, 5, 8, 17, 20 last: 0, 1, 4, 16 prior: 0
		{
			//  1*: -> (1, 0, 0)
			//  4*: -> (0, 1, 0)
			// 16*: -> (0, 0, 1)
			//  3 :                  (2, 0, 0) =  2                                                                                    -> (3, 0, 0) +(-1)
			//  6 :                  (2, 0, 0) =  2 + (1, 1, 0) =  5 - (1, 0, 0) =  1                                                  -> (2, 1, 0) +(-1, -4) -(-5)
			//  9 :                  (1, 1, 0) =  5 + (0, 2, 0) =  8 - (0, 1, 0) =  4                                                  -> (1, 2, 0) +(-1, -4) -(-5)
			// 12 :                  (0, 2, 0) =  8                                                                                    -> (0, 3, 0) +(-4)
			// 18 :                  (2, 0, 0) =  2 + (1, 0, 1) = 17 - (1, 0, 0) =  1                                                  -> (2, 0, 1) +(-1, x) -(x-1)
			// 21 : (0, 0, 0) = 0* + (1, 1, 0) =  5 + (1, 0, 1) = 17 + (0, 1, 1) = 20 - (1, 0, 0) = 1 - (0, 1, 0) = 4 - (0, 0, 1) = 16 -> (1, 1, 1) +(0*, -1, -4, x) -(-5, x-1, x-4)
			// 24 :                  (0, 2, 0) =  8 + (0, 1, 1) = 20 - (0, 1, 0) =  4                                                  -> (0, 2, 1) +(-4, x) -(x-4)
			//  1 :                  (1, 0, 1) = 17 + (0, 0, 2) =  0 - (0, 0, 1) = 16                                                  -> (1, 0, 2) +(-1, x) -(x-1)
			//  4 :                  (0, 1, 1) = 20 + (0, 0, 2) =  0 - (0, 0, 1) = 16                                                  -> (0, 1, 2) +(-4, x) -(x-4)
			// 16 :                  (0, 0, 2) =  0                                                                                    -> (0, 0, 3) +(x)
			bool first = (iter == 3);
			bool second = (iter == 1);
			int tmp_x, tmp_y, tmp_z, tmp_xy, tmp_xz, tmp_yz, tmp_xyz;
			int pred = 0;
			tmp_x = __shfl_up(last, 1);
			tmp_y = __shfl_up(last, 4);
			tmp_xy = __shfl_up(last, 5);
			// __shfl_up does not wrap
			tmp_z = __shfl_xor(last, 16);
			tmp_xz = __shfl_up(tmp_z, 1);
			tmp_yz = __shfl_up(tmp_z, 4);
			tmp_xyz = __shfl(prior, 0);
			if (d_x) pred += tmp_x;
			if (d_y) pred += tmp_y;
			if (d_x && d_y) pred -= tmp_xy;
			if (d_z || second) pred += tmp_z;
			if (d_x && (d_z || second)) pred -= tmp_xz;
			if (d_y && (d_z || second)) pred -= tmp_yz;
			if (d_x && d_y && (d_z || second)) pred += tmp_xyz;
			if (first)
			{
				last = tmp0;
			}
			else if (second)
			{
				prior = tmp0;
				last = tmp1;
			}
			if (first || second)
			{
				if (pred < min) pred = min;
				if (pred > max) pred = max;
				last = decodeDelta(last, pred, min, max);
			}
			if (first)
				tmp0 = last;
			else if (second)
				tmp1 = last;
		}
		{
			// fifth iteration
			bool first = (iter == 4);
			bool second = (iter == 2);
			if (__laneid() == 0) prior = last;
			int tmp_x, tmp_y, tmp_z, tmp_xy, tmp_xz, tmp_yz, tmp_xyz;
			int pred = 0;
			tmp_x = __shfl_up(last, 1);
			tmp_y = __shfl_up(last, 4);
			tmp_xy = __shfl_up(last, 5);
			// __shfl_up does not wrap
			tmp_z = __shfl_xor(last, 16);
			tmp_xz = __shfl_up(tmp_z, 1);
			tmp_yz = __shfl_up(tmp_z, 4);
			tmp_xyz = __shfl_xor(prior, 16);
			tmp_xyz = __shfl_up(tmp_xyz, 5);
			if (d_x) pred += tmp_x;
			if (d_y) pred += tmp_y;
			if (d_x && d_y) pred -= tmp_xy;
			if (d_z || second) pred += tmp_z;
			if (d_x && (d_z || second)) pred -= tmp_xz;
			if (d_y && (d_z || second)) pred -= tmp_yz;
			if (d_x && d_y && (d_z || second)) pred += tmp_xyz;
			if (first)
			{
				last = tmp0;
			}
			else if (second)
			{
				prior = tmp0;
				last = tmp1;
			}
			if (first || second)
			{
				if (pred < min) pred = min;
				if (pred > max) pred = max;
				last = decodeDelta(last, pred, min, max);
			}
			if (first)
				tmp0 = last;
			else if (second)
				tmp1 = last;
		}
		{
			// sixth iteration
			bool first = (iter == 5);
			bool second = (iter == 3);
			if (iter == 1) prior = last;
			int tmp_x, tmp_y, tmp_z, tmp_xy, tmp_xz, tmp_yz, tmp_xyz;
			int pred = 0;
			tmp_x = __shfl_up(last, 1);
			tmp_y = __shfl_up(last, 4);
			tmp_xy = __shfl_up(last, 5);
			// __shfl_up does not wrap
			tmp_z = __shfl_xor(last, 16);
			tmp_xz = __shfl_up(tmp_z, 1);
			tmp_yz = __shfl_up(tmp_z, 4);
			tmp_xyz = __shfl_xor(prior, 16);
			tmp_xyz = __shfl_up(tmp_xyz, 5);
			if (d_x) pred += tmp_x;
			if (d_y) pred += tmp_y;
			if (d_x && d_y) pred -= tmp_xy;
			if (d_z || second) pred += tmp_z;
			if (d_x && (d_z || second)) pred -= tmp_xz;
			if (d_y && (d_z || second)) pred -= tmp_yz;
			if (d_x && d_y && (d_z || second)) pred += tmp_xyz;
			if (first)
			{
				last = tmp0;
			}
			else if (second)
			{
				prior = tmp0;
				last = tmp1;
			}
			if (first || second)
			{
				if (pred < min) pred = min;
				if (pred > max) pred = max;
				last = decodeDelta(last, pred, min, max);
			}
			if (first)
				tmp0 = last;
			else if (second)
				tmp1 = last;
		}
		{
			// seventh iteration
			bool first = (iter == 6);
			bool second = (iter == 4);
			if (iter == 2) prior = last;
			int tmp_x, tmp_y, tmp_z, tmp_xy, tmp_xz, tmp_yz, tmp_xyz;
			int pred = 0;
			tmp_x = __shfl_up(last, 1);
			tmp_y = __shfl_up(last, 4);
			tmp_xy = __shfl_up(last, 5);
			// __shfl_up does not wrap
			tmp_z = __shfl_xor(last, 16);
			tmp_xz = __shfl_up(tmp_z, 1);
			tmp_yz = __shfl_up(tmp_z, 4);
			tmp_xyz = __shfl_xor(prior, 16);
			tmp_xyz = __shfl_up(tmp_xyz, 5);
			if (d_x) pred += tmp_x;
			if (d_y) pred += tmp_y;
			if (d_x && d_y) pred -= tmp_xy;
			if (d_z || second) pred += tmp_z;
			if (d_x && (d_z || second)) pred -= tmp_xz;
			if (d_y && (d_z || second)) pred -= tmp_yz;
			if (d_x && d_y && (d_z || second)) pred += tmp_xyz;
			if (first)
			{
				last = tmp0;
			}
			else if (second)
			{
				prior = tmp0;
				last = tmp1;
			}
			if (first || second)
			{
				if (pred < min) pred = min;
				if (pred > max) pred = max;
				last = decodeDelta(last, pred, min, max);
			}
			if (first)
				tmp0 = last;
			else if (second)
				tmp1 = last;
		}
		{
			// eighth iteration
			bool first = (iter == 7);
			bool second = (iter == 5);
			if (iter == 3) prior = last;
			int tmp_x, tmp_y, tmp_z, tmp_xy, tmp_xz, tmp_yz, tmp_xyz;
			int pred = 0;
			tmp_x = __shfl_up(last, 1);
			tmp_y = __shfl_up(last, 4);
			tmp_xy = __shfl_up(last, 5);
			// __shfl_up does not wrap
			tmp_z = __shfl_xor(last, 16);
			tmp_xz = __shfl_up(tmp_z, 1);
			tmp_yz = __shfl_up(tmp_z, 4);
			tmp_xyz = __shfl_xor(prior, 16);
			tmp_xyz = __shfl_up(tmp_xyz, 5);
			if (d_x) pred += tmp_x;
			if (d_y) pred += tmp_y;
			if (d_x && d_y) pred -= tmp_xy;
			if (d_z || second) pred += tmp_z;
			if (d_x && (d_z || second)) pred -= tmp_xz;
			if (d_y && (d_z || second)) pred -= tmp_yz;
			if (d_x && d_y && (d_z || second)) pred += tmp_xyz;
			if (first)
			{
				last = tmp0;
			}
			else if (second)
			{
				prior = tmp0;
				last = tmp1;
			}
			if (first || second)
			{
				if (pred < min) pred = min;
				if (pred > max) pred = max;
				last = decodeDelta(last, pred, min, max);
			}
			if (first)
				tmp0 = last;
			else if (second)
				tmp1 = last;
		}
		{
			// nineth iteration
			bool second = (iter == 6);
			if (iter == 4) prior = last;
			int tmp_z, tmp_xyz;
			int pred = 0;
			pred += __shfl_up(last, 1);
			pred += __shfl_up(last, 4);
			pred -= __shfl_up(last, 5);
			// __shfl_up does not wrap
			tmp_z = __shfl_xor(last, 16);
			pred += tmp_z;
			pred -= __shfl_up(tmp_z, 1);
			pred -= __shfl_up(tmp_z, 4);
			tmp_xyz = __shfl_xor(prior, 16);
			pred += __shfl_up(tmp_xyz, 5);
			if (second)
			{
				if (pred < min) pred = min;
				if (pred > max) pred = max;
				tmp1 = decodeDelta(tmp1, pred, min, max);
			}
		}
		{
			// tenth iteration (don't need to use last anymore)
			bool second = (iter == 7);
			int pred = 0;
			pred += __shfl_up(tmp1, 1);
			pred += __shfl_up(tmp1, 4);
			pred -= __shfl_up(tmp1, 5);
			pred += __shfl_up(tmp1, 16);
			pred -= __shfl_up(tmp1, 17);
			pred -= __shfl_up(tmp1, 20);
			pred += __shfl_up(tmp1, 21);
			if (second)
			{
				if (pred < min) pred = min;
				if (pred > max) pred = max;
				tmp1 = decodeDelta(tmp1, pred, min, max);
			}
		}
	}

	template <typename V, typename AT>
	__device__ __forceinline__ void decompressRBUC8x8_unpack_device(const unsigned char *comp, AT &off, int &tmp0, int &tmp1, const int &min, const int &max)
	{
		if (min == max)
		{
			// no compressed data
			tmp0 = tmp1 = min;
		}
		else
		{
			// get header
			int header = getComp(comp, off++);
			if (header == 0xff)
			{
				// uncompressed data
				tmp0 = getRaw<V>(comp, off +  __laneid()       * getInc<V>());
				tmp1 = getRaw<V>(comp, off + (__laneid() + 32) * getInc<V>());
				off += 64 * getInc<V>();
			}
			else
			{
				decompressRBUC8x8_decode_device<V>(comp, off, header, tmp0, tmp1);
				if (header < 0x40)
					decompressRBUC8x8_submin_device(tmp0, tmp1, min, max);
				else if (header < 0x80)
					decompressRBUC8x8_submax_device(tmp0, tmp1, min, max);
				else if (header < 0xc0)
					decompressRBUC8x8_haar_device(tmp0, tmp1, min, max);
				else
					decompressRBUC8x8_gradient_device(tmp0, tmp1, min, max);
			}
		}
	}

	template<typename V, typename AT>
	__device__ __forceinline__ void decompressRBUC8x8_single_device(const unsigned char *comp, AT off, V *out)
	{
		// we are using one warp (32 threads)
		// get minimum and maximum
		const int min = getRaw<V>(comp, off);
		off += getInc<V>();
		const int max = getRaw<V>(comp, off);
		off += getInc<V>();

		int tmp0, tmp1;

		decompressRBUC8x8_unpack_device<V>(comp, off, tmp0, tmp1, min, max);

		out[__laneid()]      = tmp0;
		out[__laneid() + 32] = tmp1;
	}

	template<typename V, typename S, typename AT>
	__device__ __forceinline__ void decompressRBUC8x8_vector_device(const unsigned char *comp, AT off, V *out)
	{
		// we are using one warp (32 threads)
#pragma unroll 1
		for (unsigned int i = 0; i < 4; i++)
		{
			// get minimum and maximum
			const int min = getRaw<S>(comp, off);
			off += getInc<S>();
			const int max = getRaw<S>(comp, off);
			off += getInc<S>();

			int tmp0, tmp1;

			decompressRBUC8x8_unpack_device<S>(comp, off, tmp0, tmp1, min, max);

			switch (i)
			{
			case 0:
				out[__laneid()     ].x = tmp0;
				out[__laneid() + 32].x = tmp1;
				break;
			case 1:
				out[__laneid()     ].y = tmp0;
				out[__laneid() + 32].y = tmp1;
				break;
			case 2:
				out[__laneid()     ].z = tmp0;
				out[__laneid() + 32].z = tmp1;
				break;
			default:
				out[__laneid()     ].w = tmp0;
				out[__laneid() + 32].w = tmp1;
			}
		}
	}

	template<typename V, typename AT>
	__device__ __forceinline__ void decompressRBUC8x8_device(const unsigned char *comp, AT off, V *out);

	template<>
	__device__ __forceinline__ void decompressRBUC8x8_device<unsigned char>(const unsigned char *comp, unsigned int off, unsigned char *out)
	{
		decompressRBUC8x8_single_device(comp, off, out);
	}

	template<>
	__device__ __forceinline__ void decompressRBUC8x8_device<unsigned short>(const unsigned char *comp, unsigned int off, unsigned short *out)
	{
		decompressRBUC8x8_single_device(comp, off, out);
	}

	template<>
	__device__ __forceinline__ void decompressRBUC8x8_device<uchar4>(const unsigned char *comp, unsigned int off, uchar4 *out)
	{
		decompressRBUC8x8_vector_device<uchar4, unsigned char>(comp, off, out);
	}

	template<>
	__device__ __forceinline__ void decompressRBUC8x8_device<ushort4>(const unsigned char *comp, unsigned int off, ushort4 *out)
	{
		decompressRBUC8x8_vector_device<ushort4, unsigned short>(comp, off, out);
	}

	template<>
	__device__ __forceinline__ void decompressRBUC8x8_device<unsigned char>(const unsigned char *comp, uint64 off, unsigned char *out)
	{
		decompressRBUC8x8_single_device(comp, off, out);
	}

	template<>
	__device__ __forceinline__ void decompressRBUC8x8_device<unsigned short>(const unsigned char *comp, uint64 off, unsigned short *out)
	{
		decompressRBUC8x8_single_device(comp, off, out);
	}

	template<>
	__device__ __forceinline__ void decompressRBUC8x8_device<uchar4>(const unsigned char *comp, uint64 off, uchar4 *out)
	{
		decompressRBUC8x8_vector_device<uchar4, unsigned char>(comp, off, out);
	}

	template<>
	__device__ __forceinline__ void decompressRBUC8x8_device<ushort4>(const unsigned char *comp, uint64 off, ushort4 *out)
	{
		decompressRBUC8x8_vector_device<ushort4, unsigned short>(comp, off, out);
	}
}

// always needs these two (volume type and address type) as template parameter
template <typename T>
class CompressedVolume
{
protected:
	// constructor
	unsigned char *m_comp;
public:
	// this one is allowed since we store the compacted data inside the renderer class
	CompressedVolume() { m_comp = NULL; }
	CompressedVolume(unsigned char *comp) : m_comp(comp) {}
	CompressedVolume(const CompressedVolume &a) : m_comp(a.m_comp) {}
	~CompressedVolume() { }
	void destroy() { if (m_comp != NULL) checkCudaErrors(cudaFree(m_comp)); m_comp = NULL; }

	template <typename AT>
	__device__ __forceinline__ void decompress(AT off, T *out)
	{
		CompressedVolumePrivate::decompressRBUC8x8_device<T>(m_comp, off, out);
	}
};
