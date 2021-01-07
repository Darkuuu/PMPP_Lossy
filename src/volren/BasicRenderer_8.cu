// Copyright (c) 2016 Stefan Guthe / GCC / TU-Darmstadt. All rights reserved. 
// Use of this source code is governed by the BSD 3-Clause license that can be
// found in the LICENSE file.

#include "global_defines.h"

#include "BasicRenderer.h"
#include "volumeRenderer_helper.h"

__constant__ float3 c_lDir;  // light direction
__constant__ float3x4 c_viewMatrix;  // view matrix
__constant__ float3x4 c_invViewMatrix;  // inverse view matrix

#include "BasicSampler.h"
#include "CudaRenderer.h"
#include "CudaTransferFunctions.h"

template <>
template <uint wsx, uint wsy, uint wsz>
void BasicRenderer<unsigned char>::render_internal_kernel(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH,
	float density, float brightness, float transferOffset, float transferScale, float tstep, bool white)
{
	float sampleScale = 255.0f / m_maximum;

	checkCudaErrors(cudaMemcpyToSymbol(c_lDir, &lDir, sizeof(lDir)));
	checkCudaErrors(cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix));
	checkCudaErrors(cudaMemcpyToSymbol(c_viewMatrix, viewMatrix, sizeofMatrix));

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
	if (m_lighting)
		if (m_rgba)
			if (m_components == 1)
				CudaDerivatingRenderer<wsx, wsy, wsz><<<gridSize, blockSize>>>(TextureSampler8(m_volumeTex), LightingGradientModulationTransferFunction(m_transferTex), d_output, imageW, imageH, density, brightness, transferOffset, transferScale, sampleScale, m_scale, tstep, white);
			else
				CudaRenderer<wsx, wsy, wsz><<<gridSize, blockSize>>>(TextureSampler8x4(m_volumeTex, comp_add, comp_scale), LightingGradientModulationTransferFunction(m_transferTex), d_output, imageW, imageH, density, brightness, transferOffset, transferScale, sampleScale, m_scale, tstep, white);
		else
			if (m_components == 1)
				CudaDerivatingRenderer<wsx, wsy, wsz><<<gridSize, blockSize>>>(TextureSampler8(m_volumeTex), LightingTransferFunction(m_transferTex), d_output, imageW, imageH, density, brightness, transferOffset, transferScale, sampleScale, m_scale, tstep, white);
			else
				CudaRenderer<wsx, wsy, wsz><<<gridSize, blockSize>>>(TextureSampler8x4(m_volumeTex, comp_add, comp_scale), LightingTransferFunction(m_transferTex), d_output, imageW, imageH, density, brightness, transferOffset, transferScale, sampleScale, m_scale, tstep, white);
	else
		if (m_rgba)
			if (m_components == 1)
				CudaDerivatingRenderer<wsx, wsy, wsz><<<gridSize, blockSize>>>(TextureSampler8(m_volumeTex), ColorTransferFunction(), d_output, imageW, imageH, density, brightness, transferOffset, transferScale, sampleScale, m_scale, tstep, white);
			else
				CudaRenderer<wsx, wsy, wsz><<<gridSize, blockSize>>>(TextureSampler8x4(m_volumeTex, comp_add, comp_scale), ColorTransferFunction(), d_output, imageW, imageH, density, brightness, transferOffset, transferScale, sampleScale, m_scale, tstep, white);
		else
			CudaRenderer<wsx, wsy, wsz><<<gridSize, blockSize>>>(TextureSampler8(m_volumeTex), SimpleTransferFunction(m_transferTex), d_output, imageW, imageH, density, brightness, transferOffset, transferScale, sampleScale, m_scale, tstep, white);
	checkCudaErrors(cudaDeviceSynchronize());
}

// explicit instantiation
template void BasicRenderer<uchar>::render_internal_kernel< 1,  1, 32>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
template void BasicRenderer<uchar>::render_internal_kernel< 2,  1, 16>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
template void BasicRenderer<uchar>::render_internal_kernel< 4,  1,  8>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
template void BasicRenderer<uchar>::render_internal_kernel< 8,  1,  4>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
template void BasicRenderer<uchar>::render_internal_kernel<16,  1,  2>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
template void BasicRenderer<uchar>::render_internal_kernel<32,  1,  1>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
template void BasicRenderer<uchar>::render_internal_kernel< 1,  2, 16>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
template void BasicRenderer<uchar>::render_internal_kernel< 2,  2,  8>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
template void BasicRenderer<uchar>::render_internal_kernel< 4,  2,  4>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
template void BasicRenderer<uchar>::render_internal_kernel< 8,  2,  2>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
template void BasicRenderer<uchar>::render_internal_kernel<16,  2,  1>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
template void BasicRenderer<uchar>::render_internal_kernel< 1,  4,  8>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
template void BasicRenderer<uchar>::render_internal_kernel< 2,  4,  4>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
template void BasicRenderer<uchar>::render_internal_kernel< 4,  4,  2>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
template void BasicRenderer<uchar>::render_internal_kernel< 8,  4,  1>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
template void BasicRenderer<uchar>::render_internal_kernel< 1,  8,  4>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
template void BasicRenderer<uchar>::render_internal_kernel< 2,  8,  2>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
template void BasicRenderer<uchar>::render_internal_kernel< 4,  8,  1>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
template void BasicRenderer<uchar>::render_internal_kernel< 1, 16,  2>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
template void BasicRenderer<uchar>::render_internal_kernel< 2, 16,  1>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
template void BasicRenderer<uchar>::render_internal_kernel< 1, 32,  1>(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH, float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
