// Copyright (c) 2016 Stefan Guthe / GCC / TU-Darmstadt. All rights reserved. 
// Use of this source code is governed by the BSD 3-Clause license that can be
// found in the LICENSE file.

#include "global_defines.h"

#include "CompressRBUC.h"
#include <algorithm>
#include <string.h>

// internal compression/decompression functions
template <typename T>
unsigned int compress8(T *in, unsigned char *comp)
{
	T max = T(0);
	for (unsigned int i = 0; i < 8; i++)
		max = std::max(max, in[i]);

	unsigned char bits = 0;
	while ((T(1) << bits) - T(1) < max) bits++;

	if (bits > 0u) {
		int idx = 0;
		int pos = 0;
		comp[0] = 0;
		for (int i = 0; i < 8; i++)
		{
			int rem = bits;
			comp[idx] |= (in[i] << pos);
			while (pos + rem >= 8)
			{
				idx++;
				rem -= (8 - pos);
				pos = 0;
				if (rem > 0) comp[idx] = (unsigned char)(in[i] >> (bits - rem));
				else comp[idx] = (unsigned char)0;
			}
			pos += rem;
		}
	}

	return bits;
}

template <typename T>
void decompress8(T *out, unsigned char *comp, unsigned char bits)
{
	for (int i = 0; i < 8; i++)
	{
		int idx = (i * bits) >> 3;
		int pos = (i * bits) & 7;
		T tmp(0);
		int rem = bits;
		T mask = (T(1) << bits) - T(1);
		while (rem > 0)
		{
			tmp |= ((T(comp[idx]) >> pos) << (bits - rem)) & mask;
			rem -= (8 - pos);
			idx++;
			pos = 0;
		}
		out[i] = tmp;
	}
}

// compression/decompression functions for unsigned types
template <typename T>
unsigned int compressRBUC8x8(T *in, unsigned char *comp)
{
	unsigned char bits[8];
	T comp_buffer[128];
	unsigned char *comp_data = (unsigned char *)comp_buffer;
	unsigned char comp_bits[8];
	unsigned char bits_size;
	unsigned int off = 0;
	for (unsigned int idx = 0; idx < 8; idx++)
	{
		bits[idx] = compress8<T>(&(in[idx * 8]), &(comp_data[off]));
		off += bits[idx];
	}
	bits_size = compress8<unsigned char>(bits, comp_bits);
	if (1 + bits_size + off < 1 + 64 * sizeof(T))
	{
		comp[0] = bits_size;
		if (bits_size > 0)
			memcpy(&(comp[1]), comp_bits, bits_size);
		if (off > 0)
			memcpy(&(comp[1 + bits_size]), comp_data, off);
		return 1 + bits_size + off;
	}
	else
	{
		// fallback
		comp[0] = 0xff;
		memcpy(&(comp[1]), in, 64 * sizeof(T));
		return 1 + 64 * sizeof(T);
	}
}

template <typename T>
unsigned int decompressRBUC8x8(unsigned char *comp, T *out)
{
	if (comp[0] != 0xff)
	{
		unsigned char bits[8];
		decompress8<unsigned char>(bits, &(comp[1]), comp[0]);
		unsigned int off = 1 + comp[0];
		for (unsigned int idx = 0; idx < 8; idx++)
		{
			decompress8<T>(&(out[idx * 8]), &(comp[off]), bits[idx]);
			off += bits[idx];
		}
		return off;
	}
	else
	{
		// fallback
		memcpy(out, &(comp[1]), 64 * sizeof(T));
		return 1 + 64 * sizeof(T);
	}
}

// support for signed types, float and double

template<typename T1, typename T2>
unsigned int compressRBUC8x8Signed(T1 *in, unsigned char *comp)
{
	T2 tmp[64];
	for (int i = 0; i < 64; i++)
		if (in[i] < 0)
			tmp[i] = (((T2)(~in[i])) << 1u) + 1u;
		else
			tmp[i] = ((T2)in[i]) << 1u;
	return compressRBUC8x8<T2>(tmp, comp);
}

template<typename T1, typename T2>
unsigned int compressRBUC8x8Float(T1 *in, unsigned char *comp)
{
	T2 tmp[64];
	for (int i = 0; i < 64; i++)
		if (in[i] < 0.0f)
			tmp[i] = (((T2*)in)[i] << 1u) + 1u;
		else
			tmp[i] = ((T2*)in)[i] << 1u;
	return compressRBUC8x8<T2>(tmp, comp);
}

template<typename T1, typename T2>
unsigned int decompressRBUC8x8Signed(unsigned char *comp, T1 *out)
{
	T2 tmp[64];
	unsigned int r = decompressRBUC8x8<T2>(comp, tmp);
	for (int i = 0; i < 64; i++)
		if ((tmp[i] & 1) != 0)
			out[i] = ~(T1)(tmp[i] >> 1u);
		else
			out[i] = (T1)(tmp[i] >> 1u);
	return r;
}

template<typename T1, typename T2>
unsigned int decompressRBUC8x8Float(unsigned char *comp, T1 *out)
{
	T2 tmp[64];
	unsigned int r = decompressRBUC8x8<T2>(comp, tmp);
	for (int i = 0; i < 64; i++)
		if ((tmp[i] & 1) != 0)
		{
			tmp[i] >>= 1;
			out[i] = -((T1*)tmp)[i];
		}
		else
		{
			tmp[i] >>= 1;
			out[i] = ((T1*)tmp)[i];
		}
	return r;
}

template<> unsigned int compressRBUC8x8(char *in, unsigned char *comp) { return compressRBUC8x8Signed<char, unsigned char>(in, comp); }
template<> unsigned int compressRBUC8x8(short *in, unsigned char *comp) { return compressRBUC8x8Signed<short, unsigned short>(in, comp); }
template<> unsigned int compressRBUC8x8(int *in, unsigned char *comp) { return compressRBUC8x8Signed<int, unsigned int>(in, comp); }
template<> unsigned int compressRBUC8x8(int64 *in, unsigned char *comp) { return compressRBUC8x8Signed<int64, uint64>(in, comp); }
template<> unsigned int compressRBUC8x8(float *in, unsigned char *comp) { return compressRBUC8x8Float<float, unsigned int>(in, comp); }
template<> unsigned int compressRBUC8x8(double *in, unsigned char *comp) { return compressRBUC8x8Float<double, uint64>(in, comp); }
template<> unsigned int decompressRBUC8x8(unsigned char *comp, char *out) { return decompressRBUC8x8Signed<char, unsigned char>(comp, out); }
template<> unsigned int decompressRBUC8x8(unsigned char *comp, short *out) { return decompressRBUC8x8Signed<short, unsigned short>(comp, out); }
template<> unsigned int decompressRBUC8x8(unsigned char *comp, int *out) { return decompressRBUC8x8Signed<int, unsigned int>(comp, out); }
template<> unsigned int decompressRBUC8x8(unsigned char *comp, int64 *out) { return decompressRBUC8x8Signed<int64, uint64>(comp, out); }
template<> unsigned int decompressRBUC8x8(unsigned char *comp, float *out) { return decompressRBUC8x8Float<float, unsigned int>(comp, out); }
template<> unsigned int decompressRBUC8x8(unsigned char *comp, double *out) { return decompressRBUC8x8Float<double, uint64>(comp, out); }
