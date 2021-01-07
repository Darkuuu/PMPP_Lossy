// Copyright (c) 2016 Stefan Guthe / GCC / TU-Darmstadt. All rights reserved. 
// Use of this source code is governed by the BSD 3-Clause license that can be
// found in the LICENSE file.

#include "global_defines.h"

#include "RendererRBUC8x8.h"
#include "volumeRenderer_helper.h"
#include "RendererRBUC8x8_helper.h"

extern __shared__ uint array[];
__constant__ float3 c_lDir;  // light direction

extern int bitMode;

__constant__ float3x4 c_viewMatrix;  // view matrix
__constant__ float3x4 c_invViewMatrix;  // inverse view matrix

#include "RBUC8x8Sampler.h"
#include "CudaRenderer.h"
#include "CudaTransferFunctions.h"

#define RENDER_UCHAR_M(CACHE, USAGE, OFFSET, ADR) \
	if (cacheSize == CACHE) \
		if (m_bLinearFilter) \
		{ \
			CachedVolumeLin<CACHE, unsigned char, USAGE, float, OffsetVolume<ADR>, CompressedVolume<T>, ADR, unsigned char*> cached(*((OffsetVolume<ADR>*)offsetVolume), *((CompressedVolume<T>*)compressedVolume), m_extent); \
			if (m_components == 1) \
			{ \
				RBUC8x8Sampler<decltype(cached), ADR, unsigned char *, USAGE, float> sampler(cached, OFFSET); \
				if (m_lighting) \
					if (m_rgba) \
						CudaDerivatingRenderer<wsx, wsy, wsz><<<gridSize, blockSize, numWarps * cacheSize * entrySize>>>(sampler, LightingGradientModulationTransferFunction(m_transferTex), d_output, imageW, imageH, density, brightness, transferOffset, transferScale, sampleScale, m_scale, tstep, white); \
					else \
						CudaDerivatingRenderer<wsx, wsy, wsz><<<gridSize, blockSize, numWarps * cacheSize * entrySize>>>(sampler, LightingTransferFunction(m_transferTex), d_output, imageW, imageH, density, brightness, transferOffset, transferScale, sampleScale, m_scale, tstep, white); \
				else \
					if (m_rgba) \
						CudaDerivatingRenderer<wsx, wsy, wsz><<<gridSize, blockSize, numWarps * cacheSize * entrySize>>>(sampler, ColorTransferFunction(), d_output, imageW, imageH, density, brightness, transferOffset, transferScale, sampleScale, m_scale, tstep, white); \
					else \
						CudaRenderer<wsx, wsy, wsz><<<gridSize, blockSize, numWarps * cacheSize * entrySize>>>(sampler, SimpleTransferFunction(m_transferTex), d_output, imageW, imageH, density, brightness, transferOffset, transferScale, sampleScale, m_scale, tstep, white); \
			} \
			else \
			{ \
				RBUC8x8Sampler4<decltype(cached), ADR, unsigned char *, USAGE, float4> sampler(cached, OFFSET, comp_add, comp_scale); \
				if (m_lighting) \
					if (m_rgba) \
						CudaRenderer<wsx, wsy, wsz><<<gridSize, blockSize, numWarps * cacheSize * entrySize>>>(sampler, LightingGradientModulationTransferFunction(m_transferTex), d_output, imageW, imageH, density, brightness, transferOffset, transferScale, sampleScale, m_scale, tstep, white); \
					else \
						CudaRenderer<wsx, wsy, wsz><<<gridSize, blockSize, numWarps * cacheSize * entrySize>>>(sampler, LightingTransferFunction(m_transferTex), d_output, imageW, imageH, density, brightness, transferOffset, transferScale, sampleScale, m_scale, tstep, white); \
				else \
					if (m_rgba) \
						CudaRenderer<wsx, wsy, wsz><<<gridSize, blockSize, numWarps * cacheSize * entrySize>>>(sampler, ColorTransferFunction(), d_output, imageW, imageH, density, brightness, transferOffset, transferScale, sampleScale, m_scale, tstep, white); \
					else \
						CudaRenderer<wsx, wsy, wsz><<<gridSize, blockSize, numWarps * cacheSize * entrySize>>>(sampler, DiscardColorTransferFunction(m_transferTex), d_output, imageW, imageH, density, brightness, transferOffset, transferScale, sampleScale, m_scale, tstep, white); \
			} \
		} \
		else \
		{ \
			CachedVolumeNN<CACHE, unsigned char, USAGE, float, OffsetVolume<ADR>, CompressedVolume<T>, ADR, unsigned char*> cached(*((OffsetVolume<ADR>*)offsetVolume), *((CompressedVolume<T>*)compressedVolume), m_extent); \
			if (m_components == 1) \
			{ \
				RBUC8x8Sampler<decltype(cached), ADR, unsigned char *, USAGE, float> sampler(cached, OFFSET); \
				if (m_lighting) \
					if (m_rgba) \
						CudaDerivatingRenderer<wsx, wsy, wsz><<<gridSize, blockSize, numWarps * cacheSize * entrySize>>>(sampler, LightingGradientModulationTransferFunction(m_transferTex), d_output, imageW, imageH, density, brightness, transferOffset, transferScale, sampleScale, m_scale, tstep, white); \
					else \
						CudaDerivatingRenderer<wsx, wsy, wsz><<<gridSize, blockSize, numWarps * cacheSize * entrySize>>>(sampler, LightingTransferFunction(m_transferTex), d_output, imageW, imageH, density, brightness, transferOffset, transferScale, sampleScale, m_scale, tstep, white); \
				else \
					if (m_rgba) \
						CudaDerivatingRenderer<wsx, wsy, wsz><<<gridSize, blockSize, numWarps * cacheSize * entrySize>>>(sampler, ColorTransferFunction(), d_output, imageW, imageH, density, brightness, transferOffset, transferScale, sampleScale, m_scale, tstep, white); \
					else \
						CudaRenderer<wsx, wsy, wsz><<<gridSize, blockSize, numWarps * cacheSize * entrySize>>>(sampler, SimpleTransferFunction(m_transferTex), d_output, imageW, imageH, density, brightness, transferOffset, transferScale, sampleScale, m_scale, tstep, white); \
			} \
			else \
			{ \
				RBUC8x8Sampler4<decltype(cached), ADR, unsigned char *, USAGE, float4> sampler(cached, OFFSET, comp_add, comp_scale); \
				if (m_lighting) \
					if (m_rgba) \
						CudaRenderer<wsx, wsy, wsz><<<gridSize, blockSize, numWarps * cacheSize * entrySize>>>(sampler, LightingGradientModulationTransferFunction(m_transferTex), d_output, imageW, imageH, density, brightness, transferOffset, transferScale, sampleScale, m_scale, tstep, white); \
					else \
						CudaRenderer<wsx, wsy, wsz><<<gridSize, blockSize, numWarps * cacheSize * entrySize>>>(sampler, LightingTransferFunction(m_transferTex), d_output, imageW, imageH, density, brightness, transferOffset, transferScale, sampleScale, m_scale, tstep, white); \
				else \
					if (m_rgba) \
						CudaRenderer<wsx, wsy, wsz><<<gridSize, blockSize, numWarps * cacheSize * entrySize>>>(sampler, ColorTransferFunction(), d_output, imageW, imageH, density, brightness, transferOffset, transferScale, sampleScale, m_scale, tstep, white); \
					else \
						CudaRenderer<wsx, wsy, wsz><<<gridSize, blockSize, numWarps * cacheSize * entrySize>>>(sampler, DiscardColorTransferFunction(m_transferTex), d_output, imageW, imageH, density, brightness, transferOffset, transferScale, sampleScale, m_scale, tstep, white); \
			} \
		}

template <>
template <uint wsx, uint wsy, uint wsz>
void RendererRBUC8x8<unsigned char>::render_internal_kernel(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH,
	float density, float brightness, float transferOffset, float transferScale, float tstep, bool white)
{
	float sampleScale = 255.0f / m_maximum;
	checkCudaErrors(cudaMemcpyToSymbol(c_lDir, &lDir, sizeof(lDir)));

	// 1024 threads at 64 registers per thread maximum -> 32 warps
	unsigned int numWarps = (blockSize.x * blockSize.y * blockSize.z) / 32; // numWarps = 8
	unsigned int decompSize = sizeof(unsigned char) * 64; // decompSize = 64
	unsigned int entrySize;
	if (volume64) entrySize = decompSize + sizeof(uint64); // entrySize = 72
	else entrySize = decompSize + sizeof(uint); // entrySize = 68

#ifdef KEPLER
	unsigned int cacheSize = std::min(32u, (48u * 1024u) / (entrySize * 32u)); // 22,21
#else
	unsigned int cacheSize = std::min(32u, (96u * 1024u) / (entrySize * 32u)); // 32,32
#endif
	const uint3 blkExtent = make_uint3((unsigned int)(m_extent.width + 3) / 4, (unsigned int)(m_extent.height + 3) / 4, (unsigned int)(m_extent.depth + 3) / 4);
	const float3 extentFloat = make_float3((float)m_realExtent.width, (float)m_realExtent.height, (float)m_realExtent.depth);
	const float3 extentMinusOneFloat = make_float3(m_realExtent.width - 1.0f, m_realExtent.height - 1.0f, m_realExtent.depth - 1.0f);
	float comp_add, comp_scale;
	if (m_bLinearFilter)
	{
		comp_scale = (0.25f * m_realExtent.depth - 1.0f) / (m_realExtent.depth - 1.0f);
		comp_add = (0.25f * m_realExtent.depth) / (m_realExtent.depth - 1.0f);
	}
	else
	{
		comp_scale = comp_add = 0.25f;
	}


	checkCudaErrors(cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix));
	checkCudaErrors(cudaMemcpyToSymbol(c_viewMatrix, viewMatrix, sizeofMatrix));

	if (!volume64)
	{
#ifdef KEPLER
		RENDER_UCHAR_M(22, USAGE32<22>, (8 * 22 * 64 / 4), uint)
#else
		RENDER_UCHAR_M(32, USAGE32<32>, (8 * 32 * 64 / 4), uint)
#endif
	}
	else
	{
#ifdef _WIN64
#ifdef KEPLER
		RENDER_UCHAR_M(21, USAGE32<21>, (8 * 21 * 64 / 4), uint64)
#else
		RENDER_UCHAR_M(32, USAGE32<32>, (8 * 32 * 64 / 4), uint64)
#endif
#endif
	}

	checkCudaErrors(cudaDeviceSynchronize());
}

// explicit instantiation
template void RendererRBUC8x8<uchar>::render_internal_kernel< 1,  1, 32>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
template void RendererRBUC8x8<uchar>::render_internal_kernel< 2,  1, 16>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
template void RendererRBUC8x8<uchar>::render_internal_kernel< 4,  1,  8>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
template void RendererRBUC8x8<uchar>::render_internal_kernel< 8,  1,  4>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
template void RendererRBUC8x8<uchar>::render_internal_kernel<16,  1,  2>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
template void RendererRBUC8x8<uchar>::render_internal_kernel<32,  1,  1>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
template void RendererRBUC8x8<uchar>::render_internal_kernel< 1,  2, 16>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
template void RendererRBUC8x8<uchar>::render_internal_kernel< 2,  2,  8>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
template void RendererRBUC8x8<uchar>::render_internal_kernel< 4,  2,  4>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
template void RendererRBUC8x8<uchar>::render_internal_kernel< 8,  2,  2>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
template void RendererRBUC8x8<uchar>::render_internal_kernel<16,  2,  1>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
template void RendererRBUC8x8<uchar>::render_internal_kernel< 1,  4,  8>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
template void RendererRBUC8x8<uchar>::render_internal_kernel< 2,  4,  4>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
template void RendererRBUC8x8<uchar>::render_internal_kernel< 4,  4,  2>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
template void RendererRBUC8x8<uchar>::render_internal_kernel< 8,  4,  1>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
template void RendererRBUC8x8<uchar>::render_internal_kernel< 1,  8,  4>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
template void RendererRBUC8x8<uchar>::render_internal_kernel< 2,  8,  2>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
template void RendererRBUC8x8<uchar>::render_internal_kernel< 4,  8,  1>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
template void RendererRBUC8x8<uchar>::render_internal_kernel< 1, 16,  2>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
template void RendererRBUC8x8<uchar>::render_internal_kernel< 2, 16,  1>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
template void RendererRBUC8x8<uchar>::render_internal_kernel< 1, 32,  1>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
