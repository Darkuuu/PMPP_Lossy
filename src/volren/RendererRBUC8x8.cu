// Copyright (c) 2016 Stefan Guthe / GCC / TU-Darmstadt. All rights reserved. 
// Use of this source code is governed by the BSD 3-Clause license that can be
// found in the LICENSE file.

#include <algorithm>
#include <chrono>
#include <map>

double sqerr;

#include "volumeRenderer_helper.h"
#include "RendererRBUC8x8.h"
#include "RendererRBUC8x8_helper.h"
#include <iostream>
#include <vector>

#include "OffsetVolume.h"
#include "CompressedVolume.h"

int bitMode = 2;

extern bool displayEntropy;
size_t *count;
size_t raw_bits;
extern bool g_fast;

size_t global_count[6] = { 0, 0, 0, 0, 0, 0 };
size_t global_saved[4] = { 0, 0, 0, 0 };

template <class T>
void RendererRBUC8x8<T>::setTextureFilterMode(bool bLinearFilter)
{
	m_bLinearFilter = bLinearFilter;
}

template <class T>
RendererRBUC8x8<T>::~RendererRBUC8x8()
{
	if (offsetVolume != NULL)
	{
		if (volume64)
		{
			((OffsetVolume<uint64>*)offsetVolume)->destroy();
			delete((OffsetVolume<uint64>*)offsetVolume);
		}
		else
		{
			((OffsetVolume<uint>*)offsetVolume)->destroy();
			delete((OffsetVolume<uint>*)offsetVolume);
		}
		offsetVolume = NULL;
	}
	if (compressedVolume != NULL)
	{
		// this does not have a pointer created with cudaMalloc
		//((CompressedVolume<T>*)compressedVolume)->destroy();
		delete((CompressedVolume<T>*)compressedVolume);
		compressedVolume = NULL;
	}
}

template <class T> unsigned short getVal(T& a);

template<> unsigned short getVal<unsigned char>(unsigned char &a) { return (unsigned short)a; }
template<> unsigned short getVal<unsigned short>(unsigned short &a) { return a; }
template<> unsigned short getVal<uchar4>(uchar4 &a)
{
	unsigned short r;
	r  = (unsigned short)a.x;
	r *= 17;
	r += (unsigned short)a.y;
	r *= 17;
	r += (unsigned short)a.z;
	r *= 17;
	r += (unsigned short)a.w;
	return r;
}
template<> unsigned short getVal<ushort4>(ushort4 &a)
{
	unsigned short r;
	r = a.x;
	r *= 17;
	r += a.y;
	r *= 17;
	r += a.z;
	r *= 17;
	r += a.w;
	return r;
}

template <typename T> inline float getMax(T &v);
template <> inline float getMax(unsigned char &v) { return (float)v; }
template <> inline float getMax(unsigned short &v) { return (float)v; }
template <> inline float getMax(uchar4 &v) { return (float)std::max(std::max(v.x, v.y), std::max(v.z, v.w)); }
template <> inline float getMax(ushort4 &v) { return (float)std::max(std::max(v.x, v.y), std::max(v.z, v.w)); }

template <typename T>
class PrivateBitStream
{
private:
	T pos;
	uint off;
	T* data;
	T temp;
	uint Tbits;
public:
	PrivateBitStream(T *data) : data(data) { pos = 0; off = 0; temp = 0; Tbits = sizeof(T) << 3; }
	~PrivateBitStream() {}

	void write(T value, uint bits)
	{
		T val = value & ((T(1) << bits) - T(1));
		temp |= val << off;
		off += bits;
		if (off >= Tbits)
		{
			off -= Tbits;
			data[pos++] = temp;
			temp = val >> (bits - off);
		}
	}

	void flush()
	{
		if (off > 0)
		{
			data[pos++] = temp;
			off = 0;
			temp = 0;
		}
	}

	T size()
	{
		return pos * sizeof(T);
	}
};

template <class T>
void RendererRBUC8x8<T>::initCuda(T *h_volume, cudaExtent volumeSize, cudaExtent originalSize, float3 scale, unsigned int components, int max_error)
{
	sqerr = 0.0;
	m_scale = scale;
	m_components = components;

	m_maximum = 0.0f;
	for (size_t i = 0; i < volumeSize.width * volumeSize.height * volumeSize.depth; i++) m_maximum = std::max(m_maximum, getMax(h_volume[i]));
	m_maximum = expf(logf(2.0f) * ceilf(logf(m_maximum - 1.0f) / logf(2.0f))) - 1.0f;

	// create 3D array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();

	size_t max_size = ((9 * sizeof(T) * (volumeSize.width * volumeSize.height * volumeSize.depth)) / 8);
	unsigned char *h_compressedVolume = new unsigned char[max_size];
	size_t *h_compressedVolumeUInt = new size_t[max_size >> 3];
	unsigned int r_width = (unsigned int) ((volumeSize.width + 3) >> 2);
	unsigned int r_height = (unsigned int) ((volumeSize.height + 3) >> 2);
	unsigned int r_depth =(unsigned int)  ((volumeSize.depth + 3) >> 2);
	size_t r_size = r_width * r_height * r_depth * sizeof(size_t);
	size_t c_offset = 0;
	m_extent = volumeSize;
	m_realExtent = originalSize;

	std::vector<std::vector<std::vector<T> > > ref;
	std::vector<std::vector<std::vector<float> > > vq_avg;
	std::vector<std::vector<unsigned int> > vq_count;
	std::vector<std::vector<unsigned int> > match_ref;
	std::vector<std::pair<unsigned int, unsigned int> > vq_ref;

	ref.resize(16777216);
	match_ref.resize(16777216);

	unsigned int double_ref = 0;
	unsigned int unique = 0;

	auto start_time = std::chrono::high_resolution_clock::now();

	if (displayEntropy)
	{
		count = new size_t[16777216];
		for (unsigned int i = 0; i < 16777216; i++)
			count[i] = 0;
	}

	unsigned int cur_idx = 0;
	for (unsigned int z = 0; z < r_depth; z++)
	{
		for (unsigned int y = 0; y < r_height; y++)
		{
			for (unsigned int x = 0; x < r_width; x++)
			{
				T raw_dat[64];
				std::vector<T> tmp;
				int key = 0;
				bool data_valid = false;
				if (!g_fast)
				{
					if ((z * 4 + 4 > volumeSize.depth) || (y * 4 + 4 > volumeSize.height) || (x * 4 + 4 > volumeSize.width))
					{
						unsigned int zl = std::min(4u, (unsigned int)(volumeSize.depth - (z << 2)));
						unsigned int yl = std::min(4u, (unsigned int)(volumeSize.height - (y << 2)));
						unsigned int xl = std::min(4u, (unsigned int)(volumeSize.width - (x << 2)));
						// we are at the border so do a brute-force search for a matching block to complete this one
						for (unsigned int za = 0; (za < volumeSize.depth >> 2) && (!data_valid); za++)
						{
							for (unsigned int ya = 0; (ya < volumeSize.height >> 2) && (!data_valid); ya++)
							{
								for (unsigned int xa = 0; (xa < volumeSize.depth >> 2) && (!data_valid); xa++)
								{
									data_valid = true;
									for (unsigned int z0 = 0; (z0 < zl) && data_valid; z0++)
									{
										unsigned int zr = (z << 2) + z0;
										unsigned int zc = (za << 2) + z0;
										for (unsigned int y0 = 0; (y0 < yl) && data_valid; y0++)
										{
											unsigned int yr = (y << 2) + y0;
											unsigned int yc = (ya << 2) + y0;
											size_t xr = (x << 2) + volumeSize.width * (yr + volumeSize.height * zr);
											size_t xc = (xa << 2) + volumeSize.width * (yc + volumeSize.height * zc);
											for (unsigned int x0 = 0; (x0 < xl) && data_valid; x0++)
											{
												T val = h_volume[xr++];
												T alt = h_volume[xc++];
												data_valid = (val == alt);
											}
										}
									}
									if (data_valid)
									{
										for (unsigned int z0 = 0; z0 < 4; z0++)
										{
											unsigned int zr = (za << 2) + z0;
											for (unsigned int y0 = 0; y0 < 4; y0++)
											{
												unsigned int yr = (ya << 2) + y0;
												for (unsigned int x0 = 0; x0 < 4; x0++)
												{
													unsigned int xr = (xa << 2) + x0;
													T val = h_volume[xr + volumeSize.width * (yr + volumeSize.height * zr)];
													// * Shift by 2 left = x * 4
													raw_dat[x0 + ((y0 + (z0 << 2)) << 2)] = val;
													tmp.push_back(val);
													key = (key * 13) + getVal<T>(val);
												}
											}
										}
									}
								}
							}
						}
						// now we still need to check the other borders...
					}
				}
				// check if we already filled the data
				if (!data_valid)
				{
					for (unsigned int z0 = 0; z0 < 4; z0++)
					{
						unsigned int zr = std::min((unsigned int)(volumeSize.depth - 1), (z << 2) + z0);
						for (unsigned int y0 = 0; y0 < 4; y0++)
						{
							unsigned int yr = std::min((unsigned int)(volumeSize.height - 1), (y << 2) + y0);
							for (unsigned int x0 = 0; x0 < 4; x0++)
							{
								unsigned int xr = std::min((unsigned int)(volumeSize.width - 1), (x << 2) + x0);
								T val = h_volume[xr + volumeSize.width * (yr + volumeSize.height * zr)];
								raw_dat[x0 + ((y0 + (z0 << 2)) << 2)] = val;
								tmp.push_back(val);
								key = (key * 13) + getVal<T>(val);
							}
						}
					}
				}
				// 24 bit
				key &= 0xffffff;
				// check if we got the same data already
				int match_idx = -1;
				{
					for (unsigned int i = 0; ((match_idx == -1) && (i < ref[key].size())); i++)
					{
						bool match = true;
						bool all_zero = true;
						for (unsigned int j = 0; (match && (j < tmp.size())); j++)
						{
							match = (ref[key][i][j] == tmp[j]);
							all_zero &= (tmp[j] == 0);
						}
						if (match) match_idx = match_ref[key][i];
						if (match && all_zero) global_count[4]++;
					}
				}
				unsigned int size = 0;
				{
					if (match_idx == -1)
					{
						h_compressedVolumeUInt[x + r_width * (y + r_height * z)] = c_offset;
						size = compress(raw_dat, &(h_compressedVolume[c_offset]));
					}
					if (match_idx == -1)
					{
						ref[key].push_back(tmp);
						match_ref[key].push_back(cur_idx);
						c_offset += size;
						unique++;
					}
					else
					{
						h_compressedVolumeUInt[x + r_width * (y + r_height * z)] = h_compressedVolumeUInt[match_idx];
						double_ref++;
					}
				}
				cur_idx++;
			}
		}
		std::cout << ".";
	}
	std::cout << std::endl;
	if (displayEntropy)
	{
		double ent = (double) raw_bits;
		for (unsigned int i = 0; i < 16777216; i++)
		{
			if (count[i] > 0) ent -= (double)count[i] * log((double)count[i] / (double)(volumeSize.width) / (double)(volumeSize.height) / (double)(volumeSize.depth)) / log(2.0);
		}
		printf("Encoded Entropy: %e\n", (ent / (double)(volumeSize.width) / (double)(volumeSize.height) / (double)(volumeSize.depth)));
		delete[] count;
	}

	auto end_time = std::chrono::high_resolution_clock::now();
	
	std::cout << "Gradient predictor: " << global_count[0] << " pure: " << global_saved[0] << std::endl;
	std::cout << "Haar wavelet      : " << global_count[1] << " pure: " << global_saved[1] << std::endl;
	std::cout << "Subtract maximum  : " << global_count[2] << " pure: " << global_saved[2] << std::endl;
	std::cout << "Subtract minimum  : " << global_count[3] << " pure: " << global_saved[3] << std::endl;
	std::cout << "Empty blocks      : " << global_count[4] << std::endl;
	std::cout << "Constant blocks   : " << global_count[5] << std::endl;

	std::cout << "Compression time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << "ms" << std::endl;

	size_t a_size = r_size / sizeof(size_t);
	// compact the offset volume
	int offset_indirect = 0;
	int offset_entry = 0;
	while ((size_t(1) << (unsigned int)offset_entry) <= c_offset) offset_entry++;
	while ((size_t(1) << (unsigned int)offset_indirect) < unique) offset_indirect++;
	size_t size_indirect, size_direct;
#if defined BINDEX_TAG_ID || defined INDIRECT_TAG_ID
	if ((offset_entry > 32) || (a_size > 0xffffffffull))
#else
	if (offset_entry > 32)
#endif
	{
		volume64 = true;
		size_indirect = (((offset_indirect * a_size + 63) >> 6) + ((offset_entry * unique + 63) >> 6)) << 3;
		size_direct = ((offset_entry * a_size + 63) >> 6) << 3;
	}
	else
	{
		volume64 = false;
		size_indirect = (((offset_indirect * a_size + 31) >> 5) + ((offset_entry * unique + 31) >> 5)) << 2;
		size_direct = ((offset_entry * a_size + 31) >> 5) << 2;
	}

	// for now
	//size_indirect = size_direct + 1;

	// check if direct is better
	if (size_indirect >= size_direct) offset_indirect = 0;

	unsigned char *d_volume;
	checkCudaErrors(cudaMalloc(&d_volume, std::min(size_indirect, size_direct) + c_offset));
	unsigned char *h_offsetVolume;
	h_offsetVolume = (unsigned char *)malloc(std::min(size_indirect, size_direct));

	if (offset_indirect == 0)
	{
		if (volume64)
		{
			PrivateBitStream<uint64> bs((uint64 *)h_offsetVolume);
			for (int i = 0; i < a_size; i++)
			{
				bs.write(h_compressedVolumeUInt[i], offset_entry);
			}
			bs.flush();
		}
		else
		{
			PrivateBitStream<uint> bs((uint *)h_offsetVolume);
			for (int i = 0; i < a_size; i++)
			{
				bs.write((uint)h_compressedVolumeUInt[i], offset_entry);
			}
			bs.flush();
		}
	}
	else
	{
		// indirect mode
		if (volume64)
		{
			std::map<uint64, uint64> lookup;
			uint64 ptr = 0;
			PrivateBitStream<uint64> bs((uint64 *)h_offsetVolume);
			PrivateBitStream<uint64> bs2(&(((uint64 *)h_offsetVolume)[((offset_indirect * a_size + 63) >> 6)]));
			for (int i = 0; i < a_size; i++)
			{
				auto iter = lookup.find(h_compressedVolumeUInt[i]);
				if (iter == lookup.end())
				{
					lookup[h_compressedVolumeUInt[i]] = ptr++;
					bs2.write(h_compressedVolumeUInt[i], offset_entry);
				}
				bs.write(lookup[h_compressedVolumeUInt[i]], offset_indirect);
			}
			bs.flush();
			bs2.flush();
		}
		else
		{
			std::map<uint, uint> lookup;
			uint ptr = 0;
			PrivateBitStream<uint> bs((uint *)h_offsetVolume);
			PrivateBitStream<uint> bs2(&(((uint *)h_offsetVolume)[((offset_indirect * a_size + 31) >> 5)]));
			for (int i = 0; i < a_size; i++)
			{
				auto iter = lookup.find(h_compressedVolumeUInt[i]);
				if (iter == lookup.end())
				{
					lookup[h_compressedVolumeUInt[i]] = ptr++;
					bs2.write(h_compressedVolumeUInt[i], offset_entry);
				}
				bs.write(lookup[h_compressedVolumeUInt[i]], offset_indirect);
			}
			bs.flush();
			bs2.flush();
		}
	}

	checkCudaErrors(cudaMemcpy(d_volume, h_offsetVolume, std::min(size_indirect, size_direct), cudaMemcpyHostToDevice));
	free(h_offsetVolume);

	uint3 blkExtent = make_uint3(r_width, r_height, r_depth);
	if (volume64)
		offsetVolume = new OffsetVolume<uint64>((uint64 *)d_volume, offset_entry, offset_indirect, blkExtent);
	else
		offsetVolume = new OffsetVolume<uint>((uint *)d_volume, offset_entry, offset_indirect, blkExtent);

	size_t comp_size = c_offset + std::min(size_indirect, size_direct);
	size_t orig_size = sizeof(T) * (volumeSize.width * volumeSize.height * volumeSize.depth);

	std::cout << "compression ratio (" << ((offset_indirect == 0) ? "direct" : "indirect") << ") = " << 100.0f * (float)(comp_size) / (float)(orig_size) << "% (" << comp_size << " bytes) " << double_ref << "/" << r_width * r_height * r_depth << " reused blocks" << std::endl;

	checkCudaErrors(cudaMemcpy(d_volume + std::min(size_indirect, size_direct), h_compressedVolume, c_offset, cudaMemcpyHostToDevice));
	compressedVolume = new CompressedVolume<T>(d_volume + std::min(size_indirect, size_direct));

	delete[] h_compressedVolume;
	delete[] h_compressedVolumeUInt;

	cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float4>();
	checkCudaErrors(cudaMallocArray(&d_transferFuncArray, &channelDesc2, transferFunc_size, 1));
	checkCudaErrors(cudaMemcpyToArray(d_transferFuncArray, 0, 0, transferFunc, transferFunc_size * sizeof(float4), cudaMemcpyHostToDevice));

	// create texture object
	{
		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = d_transferFuncArray;

		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.readMode = cudaReadModeElementType;
		texDesc.addressMode[0] = cudaAddressModeClamp;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.normalizedCoords = true;

		// create texture object: we only have to do this once!
		cudaCreateTextureObject(&m_transferTex, &resDesc, &texDesc, NULL);
	}

	checkCudaErrors(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
}

template <class T>
void RendererRBUC8x8<T>::updateTF()
{
	cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float4>();
	checkCudaErrors(cudaFreeArray(d_transferFuncArray));
	checkCudaErrors(cudaMallocArray(&d_transferFuncArray, &channelDesc2, transferFunc_size, 1));
	checkCudaErrors(cudaMemcpyToArray(d_transferFuncArray, 0, 0, transferFunc, transferFunc_size * sizeof(float4), cudaMemcpyHostToDevice));

	// create texture object
	{
		checkCudaErrors(cudaDestroyTextureObject(m_transferTex));

		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = d_transferFuncArray;

		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.readMode = cudaReadModeElementType;
		texDesc.addressMode[0] = cudaAddressModeClamp;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.normalizedCoords = true;

		checkCudaErrors(cudaCreateTextureObject(&m_transferTex, &resDesc, &texDesc, NULL));
	}
}

template <class T>
void RendererRBUC8x8<T>::freeCudaBuffers()
{
	if (m_transferTex != 0) checkCudaErrors(cudaDestroyTextureObject(m_transferTex));
	if (d_transferFuncArray != 0) checkCudaErrors(cudaFreeArray(d_transferFuncArray));
	d_transferFuncArray = 0;
}

template <class T>
void RendererRBUC8x8<T>::copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix)
{
	this->invViewMatrix = invViewMatrix;
	this->sizeofMatrix = sizeofMatrix;
//	checkCudaErrors(cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix));
}

template <class T>
void RendererRBUC8x8<T>::copyViewMatrix(float *viewMatrix, size_t sizeofMatrix)
{
	this->viewMatrix = viewMatrix;
	this->sizeofMatrix = sizeofMatrix;
//	checkCudaErrors(cudaMemcpyToSymbol(c_viewMatrix, viewMatrix, sizeofMatrix));
}

template <class T>
void RendererRBUC8x8<T>::render_kernel(dim3 gridSize, dim3 blockSize, dim3 warpDim, uint *d_output, uint imageW, uint imageH,
	float density, float brightness, float transferOffset, float transferScale, float tstep, bool white)
{
	switch (warpDim.x)
	{
	case 1:
		switch (warpDim.y)
		{
		case 1:
			return render_internal_kernel<1, 1, 32>(gridSize, blockSize, d_output, imageW, imageH, density, brightness, transferOffset, transferScale, tstep, white);
		case 2:
			return render_internal_kernel<1, 2, 16>(gridSize, blockSize, d_output, imageW, imageH, density, brightness, transferOffset, transferScale, tstep, white);
		case 4:
			return render_internal_kernel<1, 4, 8>(gridSize, blockSize, d_output, imageW, imageH, density, brightness, transferOffset, transferScale, tstep, white);
		case 8:
			return render_internal_kernel<1, 8, 4>(gridSize, blockSize, d_output, imageW, imageH, density, brightness, transferOffset, transferScale, tstep, white);
		case 16:
			return render_internal_kernel<1, 16, 2>(gridSize, blockSize, d_output, imageW, imageH, density, brightness, transferOffset, transferScale, tstep, white);
		default:
			return render_internal_kernel<1, 32, 1>(gridSize, blockSize, d_output, imageW, imageH, density, brightness, transferOffset, transferScale, tstep, white);
		}
	case 2:
		switch (warpDim.y)
		{
		case 1:
			return render_internal_kernel<2, 1, 16>(gridSize, blockSize, d_output, imageW, imageH, density, brightness, transferOffset, transferScale, tstep, white);
		case 2:
			return render_internal_kernel<2, 2, 8>(gridSize, blockSize, d_output, imageW, imageH, density, brightness, transferOffset, transferScale, tstep, white);
		case 4:
			return render_internal_kernel<2, 4, 4>(gridSize, blockSize, d_output, imageW, imageH, density, brightness, transferOffset, transferScale, tstep, white);
		case 8:
			return render_internal_kernel<2, 8, 2>(gridSize, blockSize, d_output, imageW, imageH, density, brightness, transferOffset, transferScale, tstep, white);
		default:
			return render_internal_kernel<2, 16, 1>(gridSize, blockSize, d_output, imageW, imageH, density, brightness, transferOffset, transferScale, tstep, white);
		}
	case 4:
		switch (warpDim.y)
		{
		case 1:
			return render_internal_kernel<4, 1, 8>(gridSize, blockSize, d_output, imageW, imageH, density, brightness, transferOffset, transferScale, tstep, white);
		case 2:
			return render_internal_kernel<4, 2, 4>(gridSize, blockSize, d_output, imageW, imageH, density, brightness, transferOffset, transferScale, tstep, white);
		case 4:
			return render_internal_kernel<4, 4, 2>(gridSize, blockSize, d_output, imageW, imageH, density, brightness, transferOffset, transferScale, tstep, white);
		default:
			return render_internal_kernel<4, 8, 1>(gridSize, blockSize, d_output, imageW, imageH, density, brightness, transferOffset, transferScale, tstep, white);
		}
	case 8:
		switch (warpDim.y)
		{
		case 1:
			return render_internal_kernel<8, 1, 4>(gridSize, blockSize, d_output, imageW, imageH, density, brightness, transferOffset, transferScale, tstep, white);
		case 2:
			return render_internal_kernel<8, 2, 2>(gridSize, blockSize, d_output, imageW, imageH, density, brightness, transferOffset, transferScale, tstep, white);
		default:
			return render_internal_kernel<8, 4, 1>(gridSize, blockSize, d_output, imageW, imageH, density, brightness, transferOffset, transferScale, tstep, white);
		}
	case 16:
		switch (warpDim.y)
		{
		case 1:
			return render_internal_kernel<16, 1, 2>(gridSize, blockSize, d_output, imageW, imageH, density, brightness, transferOffset, transferScale, tstep, white);
		default:
			return render_internal_kernel<16, 2, 1>(gridSize, blockSize, d_output, imageW, imageH, density, brightness, transferOffset, transferScale, tstep, white);
		}
	default:
		return render_internal_kernel<32, 1, 1>(gridSize, blockSize, d_output, imageW, imageH, density, brightness, transferOffset, transferScale, tstep, white);
	}
}

template <>
unsigned int RendererRBUC8x8<unsigned char>::setMinMax(unsigned char *comp, int min, int max)
{
	comp[0] = min;
	comp[1] = max;
	return 2;
}

template <>
unsigned int RendererRBUC8x8<unsigned short>::setMinMax(unsigned char *comp, int min, int max)
{
	comp[0] = (unsigned char)min;
	comp[1] = (unsigned char)(min >> 8);
	comp[2] = (unsigned char)max;
	comp[3] = (unsigned char)(max >> 8);
	return 4;
}

template <>
unsigned int RendererRBUC8x8<uchar4>::setMinMax(unsigned char *comp, int min, int max)
{
	comp[0] = min;
	comp[1] = max;
	return 2;
}

template <>
unsigned int RendererRBUC8x8<ushort4>::setMinMax(unsigned char *comp, int min, int max)
{
	comp[0] = (unsigned char)min;
	comp[1] = (unsigned char)(min >> 8);
	comp[2] = (unsigned char)max;
	comp[3] = (unsigned char)(max >> 8);
	return 4;
}

unsigned char decodeDeltaTest(unsigned char delta, unsigned char pred, unsigned char min, unsigned char max)
{
	unsigned char max_pos = max - pred;
	unsigned char max_neg = pred - min;
	// this is a corner case where the whole range is positive only
	if (max_neg == 0) return pred + delta;

	unsigned char balanced_max = __min(max_neg - 1, max_pos);
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
	if ((delta & 1) == 1)
		return pred - (delta >> 1) - 1;
	else
		return pred + (delta >> 1);
}

void transform(double &v0, double &v1)
{
	double sum = v0 + v1;
	double dif = v1 - v0;
	v0 = sqrt(0.5) * sum;
	v1 = sqrt(0.5) * dif;
}

void invTransform(double &v0, double &v1)
{
	double sum = sqrt(2.0) * v0;
	double dif = sqrt(2.0) * v1;
	v0 = 0.5 * (sum - dif);
	v1 = 0.5 * (sum + dif);
}

template <class T> T getAvg(T &a, T & b);

template <> unsigned char getAvg(unsigned char &a, unsigned char &b) { return (unsigned char)(((int)a + (int)b + 1) >> 1); }
template <> unsigned short getAvg(unsigned short &a, unsigned short &b) { return (unsigned short)(((int)a + (int)b + 1) >> 1); }
template <> uchar4 getAvg(uchar4 &a, uchar4 &b) { return make_uchar4(getAvg(a.x, b.x), getAvg(a.y, b.y), getAvg(a.z, b.z), getAvg(a.w, b.w)); }
template <> ushort4 getAvg(ushort4 &a, ushort4 &b) { return make_ushort4(getAvg(a.x, b.x), getAvg(a.y, b.y), getAvg(a.z, b.z), getAvg(a.w, b.w)); }

template <class T> T encodeInternal(T& pred, T& min, T& max, T& raw)
{
	int max_pos = (int)max - pred;
	int max_neg = (int)min - pred;
	int dlt = (int)raw - pred;
	// -max_neg = max_pos + 1 is the balanced case
	int m = (dlt < 0) ? -1 : 0;
	int balanced_max = std::min(max_neg ^ -1, max_pos);
	// balanced_max can be -1 if max_neg is 0
	if ((dlt ^ m) > balanced_max)
	{
		// off balance
		return (dlt ^ m) + (balanced_max + 1);
	}
	else
	{
		// balanced
		return (dlt << 1) ^ m;
	}
}

template <class T> T encode(T& pred, T& min, T& max, T& raw);

template<> unsigned char encode(unsigned char& pred, unsigned char& min, unsigned char& max, unsigned char& raw) { return encodeInternal<unsigned char>(pred, min, max, raw); }
template<> unsigned short encode(unsigned short& pred, unsigned short& min, unsigned short& max, unsigned short& raw) { return encodeInternal<unsigned short>(pred, min, max, raw); }
template<> uchar4 encode(uchar4& pred, uchar4& min, uchar4& max, uchar4& raw)
{
	return make_uchar4(
		encodeInternal<unsigned char>(pred.x, min.x, max.x, raw.x),
		encodeInternal<unsigned char>(pred.y, min.y, max.y, raw.y),
		encodeInternal<unsigned char>(pred.z, min.z, max.z, raw.z),
		encodeInternal<unsigned char>(pred.w, min.w, max.w, raw.w));
}
template<> ushort4 encode(ushort4& pred, ushort4& min, ushort4& max, ushort4& raw)
{
	return make_ushort4(
		encodeInternal<unsigned short>(pred.x, min.x, max.x, raw.x),
		encodeInternal<unsigned short>(pred.y, min.y, max.y, raw.y),
		encodeInternal<unsigned short>(pred.z, min.z, max.z, raw.z),
		encodeInternal<unsigned short>(pred.w, min.w, max.w, raw.w));
}

unsigned char predict(unsigned char *raw, unsigned char avg, unsigned char min, unsigned char max, unsigned int x, unsigned int y, unsigned int z, unsigned int idx)
{
	int pred = 0;
	if ((x == 0) && (y == 0) && (z == 0)) pred = avg;
	if (x > 0) pred += raw[idx - 1];
	if (y > 0) pred += raw[idx - 4];
	if ((x > 0) && (y > 0)) pred -= raw[idx - 5];
	if (z > 0) pred += raw[idx - 16];
	if ((x > 0) && (z > 0)) pred -= raw[idx - 17];
	if ((y > 0) && (z > 0)) pred -= raw[idx - 20];
	if ((x > 0) && (y > 0) && (z > 0)) pred += raw[idx - 21];
	if (pred < min) return min;
	if (pred > max) return max;
	return (unsigned char)pred;
}

unsigned short predict(unsigned short *raw, unsigned short avg, unsigned short min, unsigned short max, unsigned int x, unsigned int y, unsigned int z, unsigned int idx)
{
	int pred = 0;
	if ((x == 0) && (y == 0) && (z == 0)) pred = avg;
	if (x > 0) pred += raw[idx - 1];
	if (y > 0) pred += raw[idx - 4];
	if ((x > 0) && (y > 0)) pred -= raw[idx - 5];
	if (z > 0) pred += raw[idx - 16];
	if ((x > 0) && (z > 0)) pred -= raw[idx - 17];
	if ((y > 0) && (z > 0)) pred -= raw[idx - 20];
	if ((x > 0) && (y > 0) && (z > 0)) pred += raw[idx - 21];
	if (pred < min) return min;
	if (pred > max) return max;
	return (unsigned short)pred;
}

uchar4 predict(uchar4 *raw, uchar4 avg, uchar4 min, uchar4 max, unsigned int x, unsigned int y, unsigned int z, unsigned int idx)
{
	int4 pred = make_int4(0, 0, 0, 0);
	if ((x == 0) && (y == 0) && (z == 0)) pred = make_int4(avg.x, avg.y, avg.z, avg.w);
	if (x > 0) pred += raw[idx - 1];
	if (y > 0) pred += raw[idx - 4];
	if ((x > 0) && (y > 0)) pred -= raw[idx - 5];
	if (z > 0) pred += raw[idx - 16];
	if ((x > 0) && (z > 0)) pred -= raw[idx - 17];
	if ((y > 0) && (z > 0)) pred -= raw[idx - 20];
	if ((x > 0) && (y > 0) && (z > 0)) pred += raw[idx - 21];
	if (pred.x < min.x) pred.x = min.x;
	if (pred.y < min.y) pred.y = min.y;
	if (pred.z < min.z) pred.z = min.z;
	if (pred.w < min.w) pred.w = min.w;
	if (pred.x > max.x) pred.x = max.x;
	if (pred.y > max.y) pred.y = max.y;
	if (pred.z > max.z) pred.z = max.z;
	if (pred.w > max.w) pred.w = max.w;
	return make_uchar4(pred.x, pred.y, pred.z, pred.w);
}

ushort4 predict(ushort4 *raw, ushort4 avg, ushort4 min, ushort4 max, unsigned int x, unsigned int y, unsigned int z, unsigned int idx)
{
	int4 pred = make_int4(0, 0, 0, 0);
	if ((x == 0) && (y == 0) && (z == 0)) pred = make_int4(avg.x, avg.y, avg.z, avg.w);
	if (x > 0) pred += raw[idx - 1];
	if (y > 0) pred += raw[idx - 4];
	if ((x > 0) && (y > 0)) pred -= raw[idx - 5];
	if (z > 0) pred += raw[idx - 16];
	if ((x > 0) && (z > 0)) pred -= raw[idx - 17];
	if ((y > 0) && (z > 0)) pred -= raw[idx - 20];
	if ((x > 0) && (y > 0) && (z > 0)) pred += raw[idx - 21];
	if (pred.x < min.x) pred.x = min.x;
	if (pred.y < min.y) pred.y = min.y;
	if (pred.z < min.z) pred.z = min.z;
	if (pred.w < min.w) pred.w = min.w;
	if (pred.x > max.x) pred.x = max.x;
	if (pred.y > max.y) pred.y = max.y;
	if (pred.z > max.z) pred.z = max.z;
	if (pred.w > max.w) pred.w = max.w;
	return make_ushort4(pred.x, pred.y, pred.z, pred.w);
}

/**
* 
*/
template <class T>
void transformGradient(T *raw, T *delta, T min, T max)
{
	T avg = getAvg<T>(min, max);
	for (unsigned int z = 0; z < 4; z++)
	{
		for (unsigned int y = 0; y < 4; y++)
		{
			for (unsigned int x = 0; x < 4; x++)
			{
				unsigned int idx = x + 4 * (y + 4 * z);
				// clamping gradient predictor
				T pred = predict(raw, avg, min, max, x, y, z, idx);
				delta[idx] = encode<T>(pred, min, max, raw[idx]);
			}
		}
	}
}

template <class T, class T2>
void transformHaarInternal(T *raw, T2 *delta, T min, T max)
{
	T avg = getAvg(min, max);
	int dlt[64];
	for (unsigned int z = 0; z < 4; z++)
	{
		for (unsigned int y = 0; y < 4; y++)
		{
			for (unsigned int x = 0; x < 4; x++)
			{
				unsigned int idx = x + 4 * (y + 4 * z);
				dlt[idx] = (int)raw[idx] - avg;
			}
		}
	}
	for (unsigned int z = 0; z < 4; z++)
	{
		for (unsigned int y = 0; y < 4; y++)
		{
			for (unsigned int x = 0; x < 4; x += 2)
			{
				unsigned int idx = x + 4 * (y + 4 * z);
				dlt[idx + 1] -= dlt[idx];
				dlt[idx] += dlt[idx + 1] >> 1;
			}
		}
	}
	for (unsigned int z = 0; z < 4; z++)
	{
		for (unsigned int y = 0; y < 4; y += 2)
		{
			for (unsigned int x = 0; x < 4; x++)
			{
				unsigned int idx = x + 4 * (y + 4 * z);
				dlt[idx + 4] -= dlt[idx];
				dlt[idx] += dlt[idx + 4] >> 1;
			}
		}
	}
	for (unsigned int z = 0; z < 4; z += 2)
	{
		for (unsigned int y = 0; y < 4; y++)
		{
			for (unsigned int x = 0; x < 4; x++)
			{
				unsigned int idx = x + 4 * (y + 4 * z);
				dlt[idx + 16] -= dlt[idx];
				dlt[idx] += dlt[idx + 16] >> 1;
			}
		}
	}
	// second transform
	for (unsigned int z = 0; z < 4; z += 2)
	{
		for (unsigned int y = 0; y < 4; y += 2)
		{
			for (unsigned int x = 0; x < 4; x += 4)
			{
				unsigned int idx = x + 4 * (y + 4 * z);
				dlt[idx + 2] -= dlt[idx];
				dlt[idx] += dlt[idx + 2] >> 1;
			}
		}
	}
	for (unsigned int z = 0; z < 4; z += 2)
	{
		for (unsigned int y = 0; y < 4; y += 4)
		{
			for (unsigned int x = 0; x < 4; x += 2)
			{
				unsigned int idx = x + 4 * (y + 4 * z);
				dlt[idx + 8] -= dlt[idx];
				dlt[idx] += dlt[idx + 8] >> 1;
			}
		}
	}
	for (unsigned int z = 0; z < 4; z += 4)
	{
		for (unsigned int y = 0; y < 4; y += 2)
		{
			for (unsigned int x = 0; x < 4; x += 2)
			{
				unsigned int idx = x + 4 * (y + 4 * z);
				dlt[idx + 32] -= dlt[idx];
				dlt[idx] += dlt[idx + 32] >> 1;
			}
		}
	}
	for (unsigned int z = 0; z < 4; z++)
	{
		for (unsigned int y = 0; y < 4; y++)
		{
			for (unsigned int x = 0; x < 4; x++)
			{
				unsigned int idx = x + 4 * (y + 4 * z);
				if (dlt[idx] >= 0) delta[idx] = dlt[idx] << 1;
				else delta[idx] = (dlt[idx] << 1) ^ 0xffffffff;
			}
		}
	}
}

template <class T2>
void transformHaar(unsigned char *raw, T2 *delta, unsigned char min, unsigned char max)
{
	transformHaarInternal(raw, delta, min, max);
}

template <class T2>
void transformHaar(unsigned short *raw, T2 *delta, unsigned short min, unsigned short max)
{
	transformHaarInternal(raw, delta, min, max);
}

template <class T2>
void transformHaar(uchar4 *raw, T2 *delta, uchar4 min, uchar4 max)
{
	unsigned char raw_tmp[64];
	unsigned short delta_tmp[64];
	for (unsigned int idx = 0; idx < 64; idx++) raw_tmp[idx] = raw[idx].x;
	transformHaarInternal(raw_tmp, delta_tmp, min.x, max.x);
	for (unsigned int idx = 0; idx < 64; idx++) delta[idx].x = delta_tmp[idx];
	for (unsigned int idx = 0; idx < 64; idx++) raw_tmp[idx] = raw[idx].y;
	transformHaarInternal(raw_tmp, delta_tmp, min.y, max.y);
	for (unsigned int idx = 0; idx < 64; idx++) delta[idx].y = delta_tmp[idx];
	for (unsigned int idx = 0; idx < 64; idx++) raw_tmp[idx] = raw[idx].z;
	transformHaarInternal(raw_tmp, delta_tmp, min.z, max.z);
	for (unsigned int idx = 0; idx < 64; idx++) delta[idx].z = delta_tmp[idx];
	for (unsigned int idx = 0; idx < 64; idx++) raw_tmp[idx] = raw[idx].w;
	transformHaarInternal(raw_tmp, delta_tmp, min.w, max.w);
	for (unsigned int idx = 0; idx < 64; idx++) delta[idx].w = delta_tmp[idx];
}

template <class T2>
void transformHaar(ushort4 *raw, T2 *delta, ushort4 min, ushort4 max)
{
	unsigned short raw_tmp[64];
	unsigned int delta_tmp[64];
	for (unsigned int idx = 0; idx < 64; idx++) raw_tmp[idx] = raw[idx].x;
	transformHaarInternal(raw_tmp, delta_tmp, min.x, max.x);
	for (unsigned int idx = 0; idx < 64; idx++) delta[idx].x = delta_tmp[idx];
	for (unsigned int idx = 0; idx < 64; idx++) raw_tmp[idx] = raw[idx].y;
	transformHaarInternal(raw_tmp, delta_tmp, min.y, max.y);
	for (unsigned int idx = 0; idx < 64; idx++) delta[idx].y = delta_tmp[idx];
	for (unsigned int idx = 0; idx < 64; idx++) raw_tmp[idx] = raw[idx].z;
	transformHaarInternal(raw_tmp, delta_tmp, min.z, max.z);
	for (unsigned int idx = 0; idx < 64; idx++) delta[idx].z = delta_tmp[idx];
	for (unsigned int idx = 0; idx < 64; idx++) raw_tmp[idx] = raw[idx].w;
	transformHaarInternal(raw_tmp, delta_tmp, min.w, max.w);
	for (unsigned int idx = 0; idx < 64; idx++) delta[idx].w = delta_tmp[idx];
}

template <class T>
void transformSubMin(T *raw, T *delta, T min, T max)
{
	for (unsigned int z = 0; z < 4; z++)
	{
		for (unsigned int y = 0; y < 4; y++)
		{
			for (unsigned int x = 0; x < 4; x++)
			{
				unsigned int idx = x + 4 * (y + 4 * z);
				delta[idx] = raw[idx] - min;
			}
		}
	}
}

template <class T>
void transformSubMax(T *raw, T *delta, T min, T max)
{
	for (unsigned int z = 0; z < 4; z++)
	{
		for (unsigned int y = 0; y < 4; y++)
		{
			for (unsigned int x = 0; x < 4; x++)
			{
				unsigned int idx = x + 4 * (y + 4 * z);
				delta[idx] = max - raw[idx];
			}
		}
	}
}

template <class T>
void swizzleWavelet(T* delta, T* swizzled)
{
	const uint swizzle[64] = {
		0x00, 0x08, 0x01, 0x09,
		0x10, 0x18, 0x11, 0x19,
		0x02, 0x0a, 0x03, 0x0b,
		0x12, 0x1a, 0x13, 0x1b,

		0x20, 0x28, 0x21, 0x29,
		0x30, 0x38, 0x31, 0x39,
		0x22, 0x2a, 0x23, 0x2b,
		0x32, 0x3a, 0x33, 0x3b,

		0x04, 0x0c, 0x05, 0x0d,
		0x14, 0x1c, 0x15, 0x1d,
		0x06, 0x0e, 0x07, 0x0f,
		0x16, 0x1e, 0x17, 0x1f,

		0x24, 0x2c, 0x25, 0x2d,
		0x34, 0x3c, 0x35, 0x3d,
		0x26, 0x2e, 0x27, 0x2f,
		0x36, 0x3e, 0x37, 0x3f };
	for (unsigned int i = 0; i < 64; i++)
	{
		swizzled[swizzle[i]] = delta[i];
	}
}

template <class T>
void swizzleRegular(T* delta, T* swizzled)
{
	const uint swizzle[32] = {
		0x00, 0x01, 0x08, 0x09,
		0x02, 0x03, 0x0a, 0x0b,
		0x10, 0x11, 0x18, 0x19,
		0x12, 0x13, 0x1a, 0x1b,

		0x04, 0x05, 0x0c, 0x0d,
		0x06, 0x07, 0x0e, 0x0f,
		0x14, 0x15, 0x1c, 0x1d,
		0x16, 0x17, 0x1e, 0x1f };
	for (unsigned int i = 0; i < 64; i++)
	{
		swizzled[swizzle[i & 31] + (i & 32)] = delta[i];
	}
}

unsigned char getMin(unsigned char a, unsigned char b) { return std::min(a, b); }
unsigned short getMin(unsigned short a, unsigned short b) { return std::min(a, b); }
uchar4 getMin(uchar4 a, uchar4 b) { return make_uchar4(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z), std::min(a.w, b.w)); }
ushort4 getMin(ushort4 a, ushort4 b) { return make_ushort4(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z), std::min(a.w, b.w)); }
unsigned char getMax(unsigned char a, unsigned char b) { return std::max(a, b); }
unsigned short getMax(unsigned short a, unsigned short b) { return std::max(a, b); }
uchar4 getMax(uchar4 a, uchar4 b) { return make_uchar4(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z), std::max(a.w, b.w)); }
ushort4 getMax(ushort4 a, ushort4 b) { return make_ushort4(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z), std::max(a.w, b.w)); }

/** * unsigned char* comp is unnecessary, since it is not used in this function. 
 * Probably was done that way to match compress_internal2 parameters
 * Determines the minimum and maximum as mentioned in Lossless Paper ch. 3.3:
 * "First, we store the range of a brick as two uncompressed values min and max."
 */
template <class T>
void compress_internal1(T *raw, unsigned char *comp, T &min, T &max)
{
	min = max = raw[0];
	for (unsigned int idx = 1; idx < 64; idx++)
	{
		min = getMin(min, raw[idx]);
		max = getMax(max, raw[idx]);
	}
}

template <class T>
unsigned int compress_internal2(T *raw, unsigned char *comp, T &min, T &max)
{
	unsigned int off = 0;
	if (max != min)
	{
		unsigned char temp[4][128 * sizeof(T)];
		unsigned int size[4];
		{
			T delta[64];
			transformGradient(raw, delta, min, max);
			T swizzled[64];
			swizzleRegular(delta, swizzled);
			size[0] = compressRBUC8x8(swizzled, temp[0]);
		}
		{
			typedef typename std::conditional<std::is_same<T, unsigned char>::value || std::is_same<T, unsigned short>::value, ushort, ushort4>::type T2;
			T2 delta[64];
			transformHaar(raw, delta, min, max);
			T2 swizzled[64];
			swizzleWavelet(delta, swizzled);
			size[1] = compressRBUC8x8(swizzled, temp[1]);
		}
		{
			T delta[64];
			transformSubMax(raw, delta, min, max);
			T swizzled[64];
			swizzleRegular(delta, swizzled);
			size[2] = compressRBUC8x8(swizzled, temp[2]);
		}
		{
			T delta[64];
			transformSubMin(raw, delta, min, max);
			T swizzled[64];
			swizzleRegular(delta, swizzled);
			size[3] = compressRBUC8x8(swizzled, temp[3]);
		}
		for (unsigned int i = 0; i < 4; i++)
		{
			if (false) //if (size[i] > 64 * sizeof(T))
			{
				temp[i][0] = 0xff;
				memcpy(&(temp[i][1]), raw, 64 * sizeof(T));
				size[i] = 1 + 64 * sizeof(T);
			}
		}
		if ((size[0] < size[1]) && (size[0] < size[2]) && (size[0] < size[3]))
		{
			global_count[0]++;
			global_saved[1] += size[1] - size[0];
			global_saved[2] += size[2] - size[0];
			global_saved[3] += size[3] - size[0];
			memcpy(&(comp[off]), temp[0], size[0]);
			// mark gradient
			comp[off] |= 0xc0;
			if (displayEntropy)
			{
				raw_bits += 2 + 2 * sizeof(T);
				T delta[64];
				transformGradient(raw, delta, min, max);
				for (int i = 0; i < 64; i++) count[delta[i]]++;
			}
			return off + size[0];
		}
		else if ((size[1] < size[2]) && (size[1] < size[3]))
		{
			global_count[1]++;
			global_saved[0] += size[0] - size[1];
			global_saved[2] += size[2] - size[1];
			global_saved[3] += size[3] - size[1];
			memcpy(&(comp[off]), temp[1], size[1]);
			// mark haar
			comp[off] |= 0x80;
			if (displayEntropy)
			{
				typedef typename std::conditional<std::is_same<T, unsigned char>::value || std::is_same<T, unsigned short>::value, uint, uint4>::type T2;
				raw_bits += 2 + 2 * sizeof(T);
				T2 delta[64];
				transformHaar(raw, delta, min, max);
				for (int i = 0; i < 64; i++) count[delta[i]]++;
			}
			return off + size[1];
		}
		else if (size[2] < size[3])
		{
			global_count[2]++;
			global_saved[0] += size[0] - size[2];
			global_saved[1] += size[1] - size[2];
			global_saved[3] += size[3] - size[2];
			memcpy(&(comp[off]), temp[2], size[2]);
			// mark sub max
			comp[off] |= 0x40;
			if (displayEntropy)
			{
				raw_bits += 2 + 2 * sizeof(T);
				T delta[64];
				transformSubMax(raw, delta, min, max);
				for (int i = 0; i < 64; i++) count[delta[i]]++;
			}
			return off + size[2];
		}
		else
		{
			// fallback to uncompressed will always end up here as subtract min
			global_count[3]++;
			global_saved[0] += size[0] - size[3];
			global_saved[1] += size[1] - size[3];
			global_saved[2] += size[2] - size[3];
			memcpy(&(comp[off]), temp[3], size[3]);
			// mark sub min
			//comp[off] |= 0x00;
			if (displayEntropy)
			{
				raw_bits += 2 + 2 * sizeof(T);
				T delta[64];
				transformSubMin(raw, delta, min, max);
				for (int i = 0; i < 64; i++) count[delta[i]]++;
			}
			return off + size[3];
		}
	}
	else
	{
		if (max == 0) global_count[4]++;
		global_count[5]++;
		if (displayEntropy)
		{
			raw_bits += 2 * sizeof(T);
		}
		return off;
	}
}

template <class T>
unsigned int RendererRBUC8x8<T>::compress(T *raw, unsigned char *comp)
{
	T min, max;
	compress_internal1<T>(raw, comp, min, max);
	unsigned int off = setMinMax(comp, min, max);
	off += compress_internal2<T>(raw, &(comp[off]), min, max);
	return off;
}

template <>
unsigned int RendererRBUC8x8<uchar4>::compress(uchar4 *raw, unsigned char *comp)
{
	// override for individual storage
	unsigned char tmp[64];
	uchar4 min, max;
	compress_internal1<uchar4>(raw, comp, min, max);
	unsigned int off = setMinMax(comp, min.x, max.x);
	for (unsigned int idx = 0; idx < 64; idx++) tmp[idx] = raw[idx].x;
	off += compress_internal2<unsigned char>(&(tmp[0]), &(comp[off]), min.x, max.x);
	off += setMinMax(&(comp[off]), min.y, max.y);
	for (unsigned int idx = 0; idx < 64; idx++) tmp[idx] = raw[idx].y;
	off += compress_internal2<unsigned char>(&(tmp[0]), &(comp[off]), min.y, max.y);
	off += setMinMax(&(comp[off]), min.z, max.z);
	for (unsigned int idx = 0; idx < 64; idx++) tmp[idx] = raw[idx].z;
	off += compress_internal2<unsigned char>(&(tmp[0]), &(comp[off]), min.z, max.z);
	off += setMinMax(&(comp[off]), min.w, max.w);
	for (unsigned int idx = 0; idx < 64; idx++) tmp[idx] = raw[idx].w;
	off += compress_internal2<unsigned char>(&(tmp[0]), &(comp[off]), min.w, max.w);
	return off;
}

template <>
unsigned int RendererRBUC8x8<ushort4>::compress(ushort4 *raw, unsigned char *comp)
{
	// override for individual storage
	unsigned short tmp[64];
	ushort4 min, max;
	compress_internal1<ushort4>(raw, comp, min, max);
	unsigned int off = setMinMax(comp, min.x, max.x);
	for (unsigned int idx = 0; idx < 64; idx++) tmp[idx] = raw[idx].x;
	off += setMinMax(comp, min.y, max.y);
	off += compress_internal2<unsigned short>(&(tmp[0]), &(comp[off]), min.x, max.x);
	for (unsigned int idx = 0; idx < 64; idx++) tmp[idx] = raw[idx].y;
	off += setMinMax(comp, min.z, max.z);
	off += compress_internal2<unsigned short>(&(tmp[0]), &(comp[off]), min.y, max.y);
	for (unsigned int idx = 0; idx < 64; idx++) tmp[idx] = raw[idx].z;
	off += setMinMax(comp, min.w, max.w);
	off += compress_internal2<unsigned short>(&(tmp[0]), &(comp[off]), min.z, max.z);
	for (unsigned int idx = 0; idx < 64; idx++) tmp[idx] = raw[idx].w;
	off += compress_internal2<unsigned short>(&(tmp[0]), &(comp[off]), min.w, max.w);
	return off;
}

void BoxSplit(float3 a, float3 &d, float3 &e)
{
	if (a.x < 0.0f) d.x -= a.x;
	else e.x += a.x;
	if (a.y < 0.0f) d.y -= a.y;
	else e.y += a.y;
	if (a.z < 0.0f) d.z -= a.z;
	else e.z += a.z;
}

float BoxVol(float3 a, float3 b, float3 c)
{
	float3 d, e;
	d = e = make_float3(0.0f, 0.0f, 0.0f);
	BoxSplit(a, d, e);
	BoxSplit(b, d, e);
	BoxSplit(c, d, e);
	return (1.0f + 0.25f * (e.x + d.x)) * (1.0f + 0.25f * (e.y + d.y)) * (1.0f + 0.25f * (e.z + d.z));
}

template <class T>
void RendererRBUC8x8<T>::getLaunchParameter(dim3& warpDim, uint imageW, uint imageH, float tstep)
{
	float u = 1.0f / (float)imageW;
	float v = 1.0f / (float)imageH;

	if (imageW >= 2048) u = 1.0f / 1024.0f;
	if (imageH >= 2048) v = 1.0f / 1024.0f;

	float sampling_dist[3];
	float3 p[6];

	Ray eyeRay;

	float3x4 &h_invViewMatrix = *((float3x4 *)invViewMatrix);
	float dist_center;

	eyeRay.o = make_float3(mul(h_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
	dist_center = sqrtf(dot(eyeRay.o, eyeRay.o));

	eyeRay.d = normalize(make_float3(0.0f, 0.0f, -2.0f));
	eyeRay.d = mul(h_invViewMatrix, eyeRay.d);

	p[4] = eyeRay.o + (dist_center - 0.5f * tstep) * eyeRay.d;
	p[5] = eyeRay.o + (dist_center + 0.5f * tstep) * eyeRay.d;

	eyeRay.d = normalize(make_float3(-u, 0.0f, -2.0f));
	eyeRay.d = mul(h_invViewMatrix, eyeRay.d);

	p[0] = eyeRay.o + dist_center * eyeRay.d;

	eyeRay.d = normalize(make_float3(u, 0.0f, -2.0f));
	eyeRay.d = mul(h_invViewMatrix, eyeRay.d);

	p[1] = eyeRay.o + dist_center * eyeRay.d;

	eyeRay.d = normalize(make_float3(0.0f, -v, -2.0f));
	eyeRay.d = mul(h_invViewMatrix, eyeRay.d);

	p[2] = eyeRay.o + dist_center * eyeRay.d;

	eyeRay.d = normalize(make_float3(0.0f, v, -2.0f));
	eyeRay.d = mul(h_invViewMatrix, eyeRay.d);

	p[3] = eyeRay.o + dist_center * eyeRay.d;

	for (unsigned int i = 0; i < 6; i++) {
		p[i].x *= (float)m_realExtent.width / m_scale.x;
		p[i].y *= (float)m_realExtent.height / m_scale.y;
		if ((sizeof(T) <= 2) && (m_rgba || m_lighting) && (m_components > 1))
			p[i].z *= 0.25f * (float)m_realExtent.depth / m_scale.z;
		else
			p[i].z *= (float)m_realExtent.depth / m_scale.z;
	}

	for (unsigned int i = 0; i < 3; i++)
	{
		float3 d = p[2 * i] - p[2 * i + 1];
		sampling_dist[i] = sqrtf(dot(d, d));
	}
	sampling_dist[0] *= 0.5f;
	sampling_dist[1] *= 0.5f;

	warpDim.x = warpDim.y = warpDim.z = 1;
	if ((m_components == 1) && (m_rgba || m_lighting)) warpDim.x = warpDim.y = warpDim.z = 2;
	while (warpDim.x * warpDim.y * warpDim.z < 32)
	{
		float3 d;
		d.x = warpDim.x * sampling_dist[0];
		d.y = warpDim.y * sampling_dist[1];
		d.z = warpDim.z * sampling_dist[2];

		if ((d.x < d.y) && (d.x < d.z)) warpDim.x <<= 1;
		else if (d.y < d.z) warpDim.y <<= 1;
		else warpDim.z <<= 1;
	}
}

// valid template types
template class RendererRBUC8x8<unsigned char>;
template class RendererRBUC8x8<unsigned short>;
template class RendererRBUC8x8<uchar4>;
template class RendererRBUC8x8<ushort4>;
