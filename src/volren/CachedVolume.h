// Copyright (c) 2016 Stefan Guthe / GCC / TU-Darmstadt. All rights reserved. 
// Use of this source code is governed by the BSD 3-Clause license that can be
// found in the LICENSE file.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "CudaIntrinsics.h"


// USAGExx structs for external use as well
template <int cache>
struct USAGE16
{
	unsigned int dat;
	__device__ __forceinline__ void init() { dat = 0; }
	__device__ __forceinline__ void reinit() { dat ^= (1u << cache) - 1u; }
};

template <int cache>
struct USAGE32
{
	unsigned int dat;
	unsigned int ext;
	__device__ __forceinline__ void init() { dat = ext = 0; }
	__device__ __forceinline__ void reinit() { dat ^= 0xffffffffu >> (32 - cache); }
};

namespace CachedVolumePrivate
{
	__device__ __forceinline__ uint getBlkIndex(const uint &index, const uint &offset)
	{
		const uint x = ((index & 3) + (offset & 1)) & 3;
		const uint y = (((index >> 2) & 3) + ((offset >> 1) & 1)) & 3;
		const uint z = (((index >> 4) + (offset >> 2))) & 3;
		return x + (y << 2) + (z << 4);
	}

	__device__ __forceinline__ uint getBlkIndex0(const uint &index)
	{
		return index;
	}

	__device__ __forceinline__ uint getBlkIndex1(const uint &index)
	{
		return (index & 0x3c) + ((index + 0x01) & 0x3);
	}

	__device__ __forceinline__ uint getBlkIndex2(const uint &index)
	{
		return (index & 0x33) + ((index + 0x04) & 0xc);
	}

	__device__ __forceinline__ uint getBlkIndex3(const uint &index)
	{
		return (index & 0x30) + ((index + 0x04) & 0xc) + ((index + 0x01) & 0x3);
	}

	__device__ __forceinline__ uint getBlkIndex4(const uint &index)
	{
		return (index & 0x0f) + ((index + 0x10) & 0x30);
	}

	__device__ __forceinline__ uint getBlkIndex5(const uint &index)
	{
		return ((index + 0x10) & 0x30) + (index & 0xc) + ((index + 0x01) & 0x3);
	}

	__device__ __forceinline__ uint getBlkIndex6(const uint &index)
	{
		return ((index + 0x10) & 0x30) + ((index + 0x04) & 0xc) + (index & 0x3);
	}

	__device__ __forceinline__ uint getBlkIndex7(const uint &index)
	{
		return ((index + 0x10) & 0x30) + ((index + 0x04) & 0xc) + ((index + 0x01) & 0x3);
	}

	template <typename AT>
	__device__ __forceinline__ bool adrInvalid(AT adr)
	{
		return adr == 0xffffffff;
	}

#ifdef _WIN64
	template <>
	__device__ __forceinline__ bool adrInvalid(uint2 adr)
	{
		return adr.y == 0xffffffff;
	}
#endif

	template <typename AT>
	__device__ __forceinline__ void adrInvalidate(AT &adr)
	{
		adr = 0xffffffff;
	}

#ifdef _WIN64
	template <>
	__device__ __forceinline__ void adrInvalidate(uint2 &adr)
	{
		adr.y = 0xffffffff;
	}
#endif

	template <typename SAMPLE>
	__device__ __forceinline__ SAMPLE initSample(float v);

	template <> __device__ __forceinline__ float initSample(float v) { return v; }
	template <> __device__ __forceinline__ float4 initSample(float v) { return make_float4(v, v, v, v); }

	template <int cache>
	__device__ __forceinline__ void shuffleUsage(USAGE16<cache> &usage)
	{
		usage.dat |= __shfl_xor((int)usage.dat, 1);
		usage.dat |= __shfl_xor((int)usage.dat, 2);
		usage.dat |= __shfl_xor((int)usage.dat, 4);
		usage.dat |= __shfl_xor((int)usage.dat, 8);
		usage.dat |= __shfl_xor((int)usage.dat, 16);
		usage.dat |= (usage.dat << 16);
		usage.dat ^= (1u << cache) - 1u;
	}

	template <int cache>
	__device__ __forceinline__ void shuffleUsage(USAGE32<cache> &usage)
	{
		usage.dat |= __shfl_xor((int)usage.dat, 1);
		usage.dat |= __shfl_xor((int)usage.dat, 2);
		usage.dat |= __shfl_xor((int)usage.dat, 4);
		usage.dat |= __shfl_xor((int)usage.dat, 8);
		usage.dat |= __shfl_xor((int)usage.dat, 16);
		usage.ext |= usage.dat;
		usage.dat ^= 0xffffffffu >> (32 - cache);
	}

	template <int cache>
	__device__ __forceinline__ void updateUsage(USAGE16<cache> &usage, unsigned int &crIdx)
	{
		if (usage.dat == 0)
		{
			// swap out everything
			usage.dat = (1u << cache) - 1u;
			crIdx = 0;
		}
		else
		{
			if ((usage.dat & 0xffff) == 0)
			{
				usage.dat |= usage.dat >> 16;
				usage.dat ^= (0x10000u << cache) - 0x10000u;
			}
			crIdx = __ffs(usage.dat) - 1;
			usage.dat -= (1u << crIdx);
		}
	}

	template <int cache>
	__device__ __forceinline__ void updateUsage(USAGE32<cache> &usage, unsigned int &crIdx)
	{
		if ((usage.dat == 0) && (usage.ext == 0))
		{
			usage.dat = 0xffffffffu >> (32 - cache);
			crIdx = 0;
		}
		else
		{
			if (usage.dat == 0)
			{
				usage.dat = usage.ext;
				usage.ext ^= 0xffffffffu >> (32 - cache);
			}
			crIdx = __ffs(usage.dat) - 1;
		}
		usage.dat -= (1u << crIdx);
	}

	template <typename AT>
	__device__ __forceinline__ AT getNextAddress(const uint &msk, uint &sample_mask, const AT blk_adr[8], const unsigned int &bIdx)
	{
		const uint first = 31 - __clz(msk);
		const uint other_mask = __shfl((int)sample_mask, first);
		if (other_mask < 16)
			if (other_mask < 4)
				if (other_mask < 2)
					return __shfl(blk_adr[0], first); // 0000000x (0-1)
				else
					return __shfl(blk_adr[1], first); // 0000001x (2-3)
			else
				if (other_mask < 8)
					return __shfl(blk_adr[2], first); // 000001xx (4-7)
				else
					return __shfl(blk_adr[3], first); // 00001xxx (8-15)
		else
			if (other_mask < 64)
				if (other_mask < 32)
					return __shfl(blk_adr[4], first); // 0001xxxx (16-31)
				else
					return __shfl(blk_adr[5], first); // 001xxxxx (32-63)
			else
				if (other_mask < 128)
					return __shfl(blk_adr[6], first); // 01xxxxxx (64-127)
				else
					return __shfl(blk_adr[7], first); // 1xxxxxxx (128-255)
	}

	template <typename V, typename T> __forceinline__ __device__ V convertToSample(T &a);

	template <> __forceinline__ __device__ float4 convertToSample(uchar4 &a) { return make_float4(a.x / 255.0f, a.y / 255.0f, a.z / 255.0f, a.w / 255.0f); }
	template <> __forceinline__ __device__ float4 convertToSample(ushort4 &a) { return make_float4(a.x / 65535.0f, a.y / 65535.0f, a.z / 65535.0f, a.w / 65535.0f); }
	template <> __forceinline__ __device__ float convertToSample(unsigned char &a) { return a / 255.0f; }
	template <> __forceinline__ __device__ float convertToSample(unsigned short &a) { return a / 65535.0f; }

	template <int cache, typename AT, typename DT, typename SAMPLE, typename USAGE>
	__device__ __forceinline__ void cacheLookup(SAMPLE &sample, const AT &blk_adr, const uint &blk_idx, bool &running, AT* &decomp_adr, DT &decomp_data, const unsigned int &bIdx, USAGE &usage)
	{
		if (running)
		{
			unsigned int fIdx = 0xffffffff;
#pragma unroll
			for (unsigned int cIdx = 0; cIdx < cache; cIdx++)
			{
				AT adr = decomp_adr[bIdx + cIdx];
				if (blk_adr == adr)
				{
					fIdx = cIdx;
					break;
				}
				else if (adrInvalid(adr))
					break;
			}
			if (fIdx != 0xffffffff)
			{
				sample = convertToSample<SAMPLE>(decomp_data[(bIdx + fIdx) * 64 + blk_idx]);
				usage.dat |= 1u << fIdx;
				running = false;
			}
		}
		shuffleUsage<cache>(usage);
	}

	__device__ __forceinline__ void initFIdx(unsigned int(&fIdx)[8])
	{
		fIdx[0] = 0xffffffff;
		fIdx[1] = 0xffffffff;
		fIdx[2] = 0xffffffff;
		fIdx[3] = 0xffffffff;
		fIdx[4] = 0xffffffff;
		fIdx[5] = 0xffffffff;
		fIdx[6] = 0xffffffff;
		fIdx[7] = 0xffffffff;
	}

	template <typename AT>
	__device__ __forceinline__ void findAddress(unsigned int(&fIdx)[8], const AT &adr, uint &sample_mask, const AT(&blk_adr)[8], const unsigned int &cIdx)
	{
		if (((sample_mask & 0x01) != 0) && (blk_adr[0] == adr)) fIdx[0] = cIdx;
		if (((sample_mask & 0x02) != 0) && (blk_adr[1] == adr)) fIdx[1] = cIdx;
		if (((sample_mask & 0x04) != 0) && (blk_adr[2] == adr)) fIdx[2] = cIdx;
		if (((sample_mask & 0x08) != 0) && (blk_adr[3] == adr)) fIdx[3] = cIdx;
		if (((sample_mask & 0x10) != 0) && (blk_adr[4] == adr)) fIdx[4] = cIdx;
		if (((sample_mask & 0x20) != 0) && (blk_adr[5] == adr)) fIdx[5] = cIdx;
		if (((sample_mask & 0x40) != 0) && (blk_adr[6] == adr)) fIdx[6] = cIdx;
		if (((sample_mask & 0x80) != 0) && (blk_adr[7] == adr)) fIdx[7] = cIdx;
	}

	__forceinline__ __device__ float4 operator*(float a, float4 &b)
	{
		return make_float4(a * b.x, a * b.y, a * b.z, a * b.w);
	}

	template <typename DT, typename USAGE, typename SAMPLE>
	__device__ __forceinline__ void sampleShared(const unsigned int(&fIdx)[8], SAMPLE &sample, uint &sample_mask, const float(&w)[8], const uint &blk_idx, DT &decomp_data, const unsigned int &bIdx, USAGE &usage)
	{
		if (fIdx[0] != 0xffffffff)
		{
			sample += w[0] * convertToSample<SAMPLE>(decomp_data[(bIdx + fIdx[0]) * 64 + getBlkIndex0(blk_idx)]);
			usage.dat |= 1u << fIdx[0];
			sample_mask &= 0xfe;
		}
		if (fIdx[1] != 0xffffffff)
		{
			sample += w[1] * convertToSample<SAMPLE>(decomp_data[(bIdx + fIdx[1]) * 64 + getBlkIndex1(blk_idx)]);
			usage.dat |= 1u << fIdx[1];
			sample_mask &= 0xfd;
		}
		if (fIdx[2] != 0xffffffff)
		{
			sample += w[2] * convertToSample<SAMPLE>(decomp_data[(bIdx + fIdx[2]) * 64 + getBlkIndex2(blk_idx)]);
			usage.dat |= 1u << fIdx[2];
			sample_mask &= 0xfb;
		}
		if (fIdx[3] != 0xffffffff)
		{
			sample += w[3] * convertToSample<SAMPLE>(decomp_data[(bIdx + fIdx[3]) * 64 + getBlkIndex3(blk_idx)]);
			usage.dat |= 1u << fIdx[3];
			sample_mask &= 0xf7;
		}
		if (fIdx[4] != 0xffffffff)
		{
			sample += w[4] * convertToSample<SAMPLE>(decomp_data[(bIdx + fIdx[4]) * 64 + getBlkIndex4(blk_idx)]);
			usage.dat |= 1u << fIdx[4];
			sample_mask &= 0xef;
		}
		if (fIdx[5] != 0xffffffff)
		{
			sample += w[5] * convertToSample<SAMPLE>(decomp_data[(bIdx + fIdx[5]) * 64 + getBlkIndex5(blk_idx)]);
			usage.dat |= 1u << fIdx[5];
			sample_mask &= 0xdf;
		}
		if (fIdx[6] != 0xffffffff)
		{
			sample += w[6] * convertToSample<SAMPLE>(decomp_data[(bIdx + fIdx[6]) * 64 + getBlkIndex6(blk_idx)]);
			usage.dat |= 1u << fIdx[6];
			sample_mask &= 0xbf;
		}
		if (fIdx[7] != 0xffffffff)
		{
			sample += w[7] * convertToSample<SAMPLE>(decomp_data[(bIdx + fIdx[7]) * 64 + getBlkIndex7(blk_idx)]);
			usage.dat |= 1u << fIdx[7];
			sample_mask &= 0x7f;
		}
	}

	template <int cache, typename AT, typename DT, typename SAMPLE, typename USAGE>
	__device__ __forceinline__ void cacheLookup(SAMPLE &sample, uint &sample_mask, const float(&w)[8], const AT(&blk_adr)[8], const uint &blk_idx, bool &running, AT* &decomp_adr, DT &decomp_data, const unsigned int &bIdx, USAGE &usage)
	{
		if (running)
		{
			unsigned int fIdx[8];
			initFIdx(fIdx);
#pragma unroll
			for (unsigned int cIdx = 0; cIdx < cache; cIdx++)
			{
				AT adr = decomp_adr[bIdx + cIdx];
				if (adrInvalid(adr))
					break;
				findAddress(fIdx, adr, sample_mask, blk_adr, cIdx);
			}
			sampleShared(fIdx, sample, sample_mask, w, blk_idx, decomp_data, bIdx, usage);
			if (sample_mask == 0) running = false;
		}
		shuffleUsage<cache>(usage);
	}

	template <int cache, typename AT, typename DT, typename SAMPLE>
	__device__ __forceinline__ void cacheLookup(SAMPLE &sample, const AT &blk_adr, const uint &blk_idx, bool &running, AT* &decomp_adr, DT &decomp_data, const unsigned int &bIdx)
	{
		if (running)
		{
			unsigned int fIdx = 0xffffffff;
#pragma unroll
			for (unsigned int cIdx = 0; cIdx < cache; cIdx++)
			{
				AT adr = decomp_adr[bIdx + cIdx];
				if (blk_adr == adr)
				{
					fIdx = cIdx;
					break;
				}
				else if (adrInvalid(adr))
					break;
			}
			if (fIdx != 0xffffffff)
			{
				sample = decomp_data[(bIdx + fIdx) * 64 + blk_idx];
				running = false;
			}
		}
	}

	template <typename DT, typename SAMPLE>
	__device__ __forceinline__ void sampleShared(SAMPLE &sample, uint &sample_mask, const unsigned int(&fIdx)[8], const float(&w)[8], const uint &blk_idx, DT &decomp_data, const unsigned int &bIdx)
	{
		if (fIdx[0] != 0xffffffff)
		{
			sample += w[0] * decomp_data[(bIdx + fIdx[0]) * 64 + getBlkIndex0(blk_idx)];
			sample_mask &= 0xfe;
		}
		if (fIdx[1] != 0xffffffff)
		{
			sample += w[1] * decomp_data[(bIdx + fIdx[1]) * 64 + getBlkIndex1(blk_idx)];
			sample_mask &= 0xfd;
		}
		if (fIdx[2] != 0xffffffff)
		{
			sample += w[2] * decomp_data[(bIdx + fIdx[2]) * 64 + getBlkIndex2(blk_idx)];
			sample_mask &= 0xfb;
		}
		if (fIdx[3] != 0xffffffff)
		{
			sample += w[3] * decomp_data[(bIdx + fIdx[3]) * 64 + getBlkIndex3(blk_idx)];
			sample_mask &= 0xf7;
		}
		if (fIdx[4] != 0xffffffff)
		{
			sample += w[4] * decomp_data[(bIdx + fIdx[4]) * 64 + getBlkIndex4(blk_idx)];
			sample_mask &= 0xef;
		}
		if (fIdx[5] != 0xffffffff)
		{
			sample += w[5] * decomp_data[(bIdx + fIdx[5]) * 64 + getBlkIndex5(blk_idx)];
			sample_mask &= 0xdf;
		}
		if (fIdx[6] != 0xffffffff)
		{
			sample += w[6] * decomp_data[(bIdx + fIdx[6]) * 64 + getBlkIndex6(blk_idx)];
			sample_mask &= 0xbf;
		}
		if (fIdx[7] != 0xffffffff)
		{
			sample += w[7] * decomp_data[(bIdx + fIdx[7]) * 64 + getBlkIndex7(blk_idx)];
			sample_mask &= 0x7f;
		}
	}

	template <int cache, typename AT, typename DT, typename SAMPLE>
	__device__ __forceinline__ void cacheLookup(SAMPLE &sample, uint &sample_mask, const float(&w)[8], const AT(&blk_adr)[8], const uint &blk_idx, bool &running, AT* &decomp_adr, DT &decomp_data, const unsigned int &bIdx)
	{
		if (running)
		{
			unsigned int fIdx[8];
			initFIdx(fIdx);
#pragma unroll
			for (unsigned int cIdx = 0; cIdx < cache; cIdx++)
			{
				AT adr = decomp_adr[bIdx + cIdx];
				if (adrInvalid(adr))
					break;
				findAddress(fIdx, adr, sample_mask, blk_adr, cIdx);
			}
			sampleShared(sample, sample_mask, fIdx, w, blk_idx, decomp_data, bIdx);
			if (sample_mask == 0) running = false;
		}
	}

	template <typename DT, typename AT, typename SAMPLE>
	__device__ __forceinline__ void cacheLookupQuick(SAMPLE &sample, const AT &blk_adr, const AT &adr, const uint &blk_idx, const unsigned int &crIdx, bool &running, DT &decomp_data, const unsigned int &bIdx)
	{
		if (running)
		{
			if (blk_adr == adr)
			{
				// unsigned short is actually 12 bit only
				sample = convertToSample<SAMPLE>(decomp_data[(bIdx + crIdx) * 64 + blk_idx]);
				running = false;
			}
		}
	}

	template <typename DT, typename AT, typename SAMPLE>
	__device__ __forceinline__ void cacheLookupQuick(SAMPLE &sample, unsigned int &sample_mask, const float(&w)[8], const AT(&blk_adr)[8], const AT &adr, const uint &blk_idx, const unsigned int &crIdx, bool &running, DT &decomp_data, const unsigned int &bIdx)
	{
		if (running)
		{
			if (blk_adr[0] == adr)
			{
				sample += w[0] * convertToSample<SAMPLE>(decomp_data[(bIdx + crIdx) * 64 + getBlkIndex0(blk_idx)]);
				sample_mask &= 0xfe;
			}
			if (blk_adr[1] == adr)
			{
				sample += w[1] * convertToSample<SAMPLE>(decomp_data[(bIdx + crIdx) * 64 + getBlkIndex1(blk_idx)]);
				sample_mask &= 0xfd;
			}
			if (blk_adr[2] == adr)
			{
				sample += w[2] * convertToSample<SAMPLE>(decomp_data[(bIdx + crIdx) * 64 + getBlkIndex2(blk_idx)]);
				sample_mask &= 0xfb;
			}
			if (blk_adr[3] == adr)
			{
				sample += w[3] * convertToSample<SAMPLE>(decomp_data[(bIdx + crIdx) * 64 + getBlkIndex3(blk_idx)]);
				sample_mask &= 0xf7;
			}
			if (blk_adr[4] == adr)
			{
				sample += w[4] * convertToSample<SAMPLE>(decomp_data[(bIdx + crIdx) * 64 + getBlkIndex4(blk_idx)]);
				sample_mask &= 0xef;
			}
			if (blk_adr[5] == adr)
			{
				sample += w[5] * convertToSample<SAMPLE>(decomp_data[(bIdx + crIdx) * 64 + getBlkIndex5(blk_idx)]);
				sample_mask &= 0xdf;
			}
			if (blk_adr[6] == adr)
			{
				sample += w[6] * convertToSample<SAMPLE>(decomp_data[(bIdx + crIdx) * 64 + getBlkIndex6(blk_idx)]);
				sample_mask &= 0xbf;
			}
			if (blk_adr[7] == adr)
			{
				sample += w[7] * convertToSample<SAMPLE>(decomp_data[(bIdx + crIdx) * 64 + getBlkIndex7(blk_idx)]);
				sample_mask &= 0x7f;
			}
			if (sample_mask == 0) running = false;
		}
	}

	template <int cache, typename AT, typename USAGE>
	__device__ __forceinline__ AT cacheReplace(const uint &msk, const AT &blk_adr, USAGE &usage, AT* &decomp_adr, unsigned int &crIdx, const unsigned int &bIdx)
	{
		updateUsage<cache>(usage, crIdx);
		const uint first = 31 - __clz(msk);
		const AT adr = __shfl(blk_adr, first);
		if (__laneid() == 0)
		{
			// remember new location
			decomp_adr[bIdx + crIdx] = adr;
		}
		return adr;
	}

	template <int cache, typename AT, typename USAGE>
	__device__ __forceinline__ AT cacheReplace(const uint &msk, uint &sample_mask, const AT blk_adr[8], USAGE &usage, AT* &decomp_adr, unsigned int &crIdx, const unsigned int &bIdx)
	{
		updateUsage<cache>(usage, crIdx);
		AT adr = getNextAddress(msk, sample_mask, blk_adr, bIdx);
		if (__laneid() == 0)
		{
			// remember new location
			decomp_adr[bIdx + crIdx] = adr;
		}
		return adr;
	}

	template <int cache, typename T, typename USAGE, typename SAMPLE, typename OV, typename CV, typename AT, typename DT>
	__device__ __forceinline__ SAMPLE sampleCompressedNN(const float3 &pos, const float3 &extent, const float3 &extentMinusOne, bool running, const uint3 &blkExtent, OV &offsetVolume, CV &compressedVolume, AT* &decomp_adr, DT &decomp_data, USAGE &usage, const unsigned int &bIdx)
	{
		// make sure we keep the cache size small
		unsigned int crIdx = 0;
		// linear interpolation requires multiplication by extent-1, nn requires mul. by extent. Floor for nn only!
		const uint3 coord = make_uint3(
			(int)floor(__min(pos.x * extent.x, extentMinusOne.x)),
			(int)floor(__min(pos.y * extent.y, extentMinusOne.y)),
			(int)floor(__min(pos.z * extent.z, extentMinusOne.z)));
		const uint3 blk_coord = make_uint3(coord.x >> 2, coord.y >> 2, coord.z >> 2);
		if ((blk_coord.x >= blkExtent.x) || (blk_coord.y >= blkExtent.y) || (blk_coord.z >= blkExtent.z)) running = false;
		SAMPLE sample = initSample<SAMPLE>(0.0f);
		if (!__any(running)) return sample;

#ifdef ADDRESS_TAG_ID
		const AT blk_adr = offsetVolume.getAddress(blk_coord);
#endif
#if defined BINDEX_TAG_ID || defined INDIRECT_TAG_ID
		const AT blk_adr = offsetVolume.getTag(blk_coord);
#endif
		const uint blk_idx = (coord.x & 3) + (coord.y & 3) * 4 + (coord.z & 3) * 16;

		if (__any(running))
		{
			cacheLookup<cache>(sample, blk_adr, blk_idx, running, decomp_adr, decomp_data, bIdx, usage);
		}

		uint msk;
		while ((msk = __ballot(running)) != 0)
		{
			// find the next one to replace (last chance cache replacement aka. LRU with current usage)
			const AT adr = cacheReplace<cache>(msk, blk_adr, usage, decomp_adr, crIdx, bIdx);

			// decompression
#ifdef ADDRESS_TAG_ID
			compressedVolume.decompress(adr, &(decomp_data[(bIdx + crIdx) * 64]));
#endif
#if defined BINDEX_TAG_ID || defined INDIRECT_TAG_ID
			AT real_adr = offsetVolume.getAddress(adr);
			compressedVolume.decompress(real_adr, &(decomp_data[(bIdx + crIdx) * 64]));
#endif

			// get data for all threads matching the newly replaced block
			cacheLookupQuick(sample, blk_adr, adr, blk_idx, crIdx, running, decomp_data, bIdx);

		}
		return sample;
	}

	template <int cache, typename T, typename USAGE, typename SAMPLE, typename OV, typename CV, typename AT, typename DT>
	__device__ __forceinline__ SAMPLE sampleCompressedLin(const float3 &pos, const float3 &extentMinusOne, bool running, const uint3 &blkExtent, OV &offsetVolume, CV &compressedVolume, AT* &decomp_adr, DT &decomp_data, USAGE &usage, const unsigned int &bIdx)
	{
		// make sure we keep the cache size small
		unsigned int crIdx = 0;
		// linear interpolation requires multiplication by extent-1, nn requires mul. by extent. Floor for nn only!
		const float3 fcoord = make_float3(
			__min(pos.x * extentMinusOne.x, extentMinusOne.x),
			__min(pos.y * extentMinusOne.y, extentMinusOne.y),
			__min(pos.z * extentMinusOne.z, extentMinusOne.z));
		AT blk_adr[8];
		uint blk_idx;
		float w[8];

		const float3 coord0 = make_float3(floor(fcoord.x), floor(fcoord.y), floor(fcoord.z));
		const uint3 blk_coord0 = make_uint3(((uint)coord0.x) >> 2, ((uint)coord0.y) >> 2, ((uint)coord0.z) >> 2);
		const uint3 blk_coord1 = make_uint3(((uint)ceil(fcoord.x)) >> 2, ((uint)ceil(fcoord.y)) >> 2, ((uint)ceil(fcoord.z)) >> 2);
		blk_idx = (((uint)coord0.x) & 3) + ((((uint)coord0.y) & 3) << 2) + ((((uint)coord0.z) & 3) << 4);

		{
			const float alpha = fcoord.x - coord0.x;
			w[0] = 1.0f - alpha;
			w[1] = alpha;
		}

		{
			const float alpha = fcoord.y - coord0.y;
			w[2] = w[0] * alpha;
			w[3] = w[1] * alpha;
			w[0] -= w[2];
			w[1] -= w[3];
		}

		{
			const float alpha = fcoord.z - coord0.z;
			w[4] = w[0] * alpha;
			w[5] = w[1] * alpha;
			w[6] = w[2] * alpha;
			w[7] = w[3] * alpha;
			w[0] -= w[4];
			w[1] -= w[5];
			w[2] -= w[6];
			w[3] -= w[7];
		}

#ifdef ADDRESS_TAG_ID
		blk_adr[0] = offsetVolume.getAddress(blk_coord0);
		blk_adr[1] = offsetVolume.getAddress(make_uint3(blk_coord1.x, blk_coord0.y, blk_coord0.z));
		blk_adr[2] = offsetVolume.getAddress(make_uint3(blk_coord0.x, blk_coord1.y, blk_coord0.z));
		blk_adr[3] = offsetVolume.getAddress(make_uint3(blk_coord1.x, blk_coord1.y, blk_coord0.z));
		blk_adr[4] = offsetVolume.getAddress(make_uint3(blk_coord0.x, blk_coord0.y, blk_coord1.z));
		blk_adr[5] = offsetVolume.getAddress(make_uint3(blk_coord1.x, blk_coord0.y, blk_coord1.z));
		blk_adr[6] = offsetVolume.getAddress(make_uint3(blk_coord0.x, blk_coord1.y, blk_coord1.z));
		blk_adr[7] = offsetVolume.getAddress(blk_coord1);
#endif
#if defined BINDEX_TAG_ID || defined INDIRECT_TAG_ID
		blk_adr[0] = offsetVolume.getTag(blk_coord0);
		blk_adr[1] = offsetVolume.getTag(make_uint3(blk_coord1.x, blk_coord0.y, blk_coord0.z));
		blk_adr[2] = offsetVolume.getTag(make_uint3(blk_coord0.x, blk_coord1.y, blk_coord0.z));
		blk_adr[3] = offsetVolume.getTag(make_uint3(blk_coord1.x, blk_coord1.y, blk_coord0.z));
		blk_adr[4] = offsetVolume.getTag(make_uint3(blk_coord0.x, blk_coord0.y, blk_coord1.z));
		blk_adr[5] = offsetVolume.getTag(make_uint3(blk_coord1.x, blk_coord0.y, blk_coord1.z));
		blk_adr[6] = offsetVolume.getTag(make_uint3(blk_coord0.x, blk_coord1.y, blk_coord1.z));
		blk_adr[7] = offsetVolume.getTag(blk_coord1);
#endif

		// only try to read for running threads
		SAMPLE sample = initSample<SAMPLE>(0.0f);
		unsigned int sample_mask = 0xff;

		// loop until we are done
		if (__any(running))
		{
			cacheLookup<cache>(sample, sample_mask, w, blk_adr, blk_idx, running, decomp_adr, decomp_data, bIdx, usage);
		}

		uint msk;
		while ((msk = __ballot(running)) != 0)
		{
			// find the next one to replace (last chance cache replacement aka. LRU with current usage)
			const AT adr = cacheReplace<cache>(msk, sample_mask, blk_adr, usage, decomp_adr, crIdx, bIdx);

			// decompression
#ifdef ADDRESS_TAG_ID
			compressedVolume.decompress(adr, &(decomp_data[(bIdx + crIdx) * 64]));
#endif
#if defined BINDEX_TAG_ID || defined INDIRECT_TAG_ID
			AT real_adr = offsetVolume.getAddress(adr);
			compressedVolume.decompress(real_adr, &(decomp_data[(bIdx + crIdx) * 64]));
#endif

			// get data for all threads matching the newly replaced block
			cacheLookupQuick(sample, sample_mask, w, blk_adr, adr, blk_idx, crIdx, running, decomp_data, bIdx);

		}
		return sample;
	}
}

// this one does not offer interpolation
template <int cache, typename T, typename USAGE, typename SAMPLE, typename OV, typename CV, typename AT, typename DT>
class CachedVolumeNN
{
protected:
	// init
	AT *m_decomp_adr;
	DT m_decomp_data;
	unsigned int m_bIdx;

	// constructor
	float3 m_extent, m_extentMinusOne;
	uint3 m_blkExtent;
	OV m_offsetVolume;
	CV m_compressedVolume;
private:
	CachedVolumeNN();
public:
	CachedVolumeNN(OV offsetVolume, CV compressedVolume, cudaExtent extent) : m_offsetVolume(offsetVolume), m_compressedVolume(compressedVolume)
	{
		// these constants are calculated here to avoid doing this in the kernel
		m_extent.x = (float)extent.width;
		m_extent.y = (float)extent.height;
		m_extent.z = (float)extent.depth;
		m_extentMinusOne.x = (float)(extent.width - 1);
		m_extentMinusOne.y = (float)(extent.height - 1);
		m_extentMinusOne.z = (float)(extent.depth - 1);
		m_blkExtent.x = (int)((extent.width + 3) >> 2);
		m_blkExtent.y = (int)((extent.height + 3) >> 2);
		m_blkExtent.z = (int)((extent.depth + 3) >> 2);
	}
	CachedVolumeNN(const CachedVolumeNN& a) : m_offsetVolume(a.m_offsetVolume), m_compressedVolume(a.m_compressedVolume),
		m_extent(a.m_extent), m_extentMinusOne(a.m_extentMinusOne), m_blkExtent(a.m_blkExtent) {}
	~CachedVolumeNN() {}

	__device__ __forceinline__ void init(AT *decomp_adr, DT decomp_data)
	{
		m_bIdx = (threadIdx.y + threadIdx.z * blockDim.y) * cache;

		m_decomp_adr = decomp_adr;
		m_decomp_data = decomp_data;

		if (__laneid() < cache)
			CachedVolumePrivate::adrInvalidate(decomp_adr[m_bIdx + __laneid()]);

		// make sure shared memory is broadcasted
		__threadfence_block();
	}

	__device__ __forceinline__ SAMPLE sample(bool running, const float3 &pos)
	{
		USAGE usage;
		usage.init();
		return CachedVolumePrivate::sampleCompressedNN<cache, T, USAGE, SAMPLE, OV, CV, AT, DT>(pos, m_extent, m_extentMinusOne, running, m_blkExtent, m_offsetVolume, m_compressedVolume, m_decomp_adr, m_decomp_data, usage, m_bIdx);
	}
};

// this one does tri-linear interpolation
template <int cache, typename T, typename USAGE, typename SAMPLE, typename OV, typename CV, typename AT, typename DT>
class CachedVolumeLin
{
protected:
	// init
	AT *m_decomp_adr;
	DT m_decomp_data;
	unsigned int m_bIdx;

	// constructor
	float3 m_extentMinusOne;
	uint3 m_blkExtent;
	OV m_offsetVolume;
	CV m_compressedVolume;
private:
	CachedVolumeLin() {}
public:
	CachedVolumeLin(const OV& offsetVolume, const CV& compressedVolume, const cudaExtent &extent) : m_offsetVolume(offsetVolume), m_compressedVolume(compressedVolume)
	{
		// these constants are calculated here to avoid doing this in the kernel
		m_extentMinusOne.x = (float)(extent.width - 1);
		m_extentMinusOne.y = (float)(extent.height - 1);
		m_extentMinusOne.z = (float)(extent.depth - 1);
		m_blkExtent.x = (uint) ((extent.width + 3) >> 2);
		m_blkExtent.y = (uint) ((extent.height + 3) >> 2);
		m_blkExtent.z = (uint) ((extent.depth + 3) >> 2);
	}
	CachedVolumeLin(const CachedVolumeLin& a) : m_offsetVolume(a.m_offsetVolume), m_compressedVolume(a.m_compressedVolume),
		m_extentMinusOne(a.m_extentMinusOne), m_blkExtent(a.m_blkExtent) {}
	~CachedVolumeLin() {}

	__device__ __forceinline__ void init(AT *decomp_adr, DT decomp_data)
	{
		m_bIdx = (threadIdx.y + threadIdx.z * blockDim.y) * cache;

		m_decomp_adr = decomp_adr;
		m_decomp_data = decomp_data;

		if (__laneid() < cache)
			CachedVolumePrivate::adrInvalidate(decomp_adr[m_bIdx + __laneid()]);

		// make sure shared memory is broadcasted
		__threadfence_block();
	}

	__device__ __forceinline__ SAMPLE sample(bool running, const float3 &pos)
	{
		USAGE usage;
		usage.init();
		return CachedVolumePrivate::sampleCompressedLin<cache, T, USAGE, SAMPLE, OV, CV, AT, DT>(pos, m_extentMinusOne, running, m_blkExtent, m_offsetVolume, m_compressedVolume, m_decomp_adr, m_decomp_data, usage, m_bIdx);
	}
};

// this one offers optional tri-linear interpolation
template <int cache, typename T, typename USAGE, typename SAMPLE, typename OV, typename CV, typename AT, typename DT>
class CachedVolume
{
protected:
	// init
	AT *m_decomp_adr;
	DT m_decomp_data;
	unsigned int m_bIdx;

	// constructor
	float3 m_extent, m_extentMinusOne;
	uint3 m_blkExtent;
	OV m_offsetVolume;
	CV m_compressedVolume;
	bool m_interpolate;
private:
	CachedVolume() {}
public:
	CachedVolume(OV offsetVolume, CV compressedVolume, cudaExtent extent, bool interpolate) : m_offsetVolume(offsetVolume), m_compressedVolume(compressedVolume), m_interpolate(interpolate)
	{
		// these constants are calculated here to avoid doing this in the kernel
		m_extent.x = (float)extent.width;
		m_extent.y = (float)extent.height;
		m_extent.z = (float)extent.depth;
		m_extentMinusOne.x = (float)(extent.width - 1);
		m_extentMinusOne.y = (float)(extent.height - 1);
		m_extentMinusOne.z = (float)(extent.depth - 1);
		m_blkExtent.x = (extent.width + 3) >> 2;
		m_blkExtent.y = (extent.height + 3) >> 2;
		m_blkExtent.z = (extent.depth + 3) >> 2;
	}
	CachedVolume(const CachedVolume& a) : m_offsetVolume(a.m_offsetVolume), m_compressedVolume(a.m_compressedVolume), m_interpolate(a.m_interpolate), 
		m_extent(a.m_extent), m_extentMinusOne(a.m_extentMinusOne), m_blkExtent(a.m_blkExtent) {}
	~CachedVolume() {}

	// maybe someone wants to do this on the device as well.
	__host__ __device__ __forceinline__ void setInterpolation(bool interpolate) { m_interpolate = interpolate; }
	__host__ __device__ __forceinline__ bool getInterpolation() { return m_interpolate; }

	__device__ __forceinline__ void init(AT *decomp_adr, DT decomp_data)
	{
		m_bIdx = (threadIdx.y + threadIdx.z * blockDim.y) * cache;

		m_decomp_adr = decomp_adr;
		m_decomp_data = decomp_data;

		if (__laneid() < cache)
			CachedVolumePrivate::adrInvalidate(decomp_adr[m_bIdx + __laneid()]);

		// make sure shared memory is broadcasted
		__threadfence_block();
	}

	__device__ __forceinline__ SAMPLE sample(bool running, const float3 &pos, USAGE &usage)
	{
		if (m_interpolate)
			return CachedVolumePrivate::sampleCompressedLin<cache, T, USAGE, SAMPLE, OV, CV, AT, DT>(pos, m_extentMinusOne, running, m_blkExtent, m_offsetVolume, m_compressedVolume, m_decomp_adr, m_decomp_data, usage, m_bIdx);
		else
			return CachedVolumePrivate::sampleCompressedNN<cache, T, USAGE, SAMPLE, OV, CV, AT, DT>(pos, m_extent, m_extentMinusOne, running, m_blkExtent, m_offsetVolume, m_compressedVolume, m_decomp_adr, m_decomp_data, usage, &m_bIdx);
	}
};

