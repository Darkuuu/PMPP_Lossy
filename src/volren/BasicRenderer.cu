// Copyright (c) 2016 Stefan Guthe / GCC / TU-Darmstadt. All rights reserved. 
// Use of this source code is governed by the BSD 3-Clause license that can be
// found in the LICENSE file.

// Simple 3D volume renderer

#include "volumeRenderer_helper.h"
#include "BasicRenderer.h"

#include <algorithm>
#include <iostream>

__constant__ float3x4 c_viewMatrix;  // view matrix
__constant__ float3x4 c_invViewMatrix;  // inverse view matrix

__constant__ float3 c_lDir;  // light direction

#include "CudaRenderer.h"
#include "CudaTransferFunctions.h"
#include "BasicSampler.h"

template <class T>
void BasicRenderer<T>::setTextureFilterMode(bool bLinearFilter)
{
	// recreate texture object with different filter mode

	cudaResourceDesc resDesc;
	cudaGetTextureObjectResourceDesc(&resDesc, m_volumeTex);
	cudaTextureDesc texDesc;
	cudaGetTextureObjectTextureDesc(&texDesc, m_volumeTex);
	checkCudaErrors(cudaDestroyTextureObject(m_volumeTex));

	texDesc.filterMode = bLinearFilter ? cudaFilterModeLinear : cudaFilterModePoint;
	checkCudaErrors(cudaCreateTextureObject(&m_volumeTex, &resDesc, &texDesc, NULL));

	m_bLinearFilter = bLinearFilter;
}

template <typename T> inline float getMax(T &v);
template <> inline float getMax(unsigned char &v) { return (float)v; }
template <> inline float getMax(unsigned short &v) { return (float)v; }
template <> inline float getMax(uchar4 &v) { return (float)std::max(std::max(v.x, v.y), std::max(v.z, v.w)); }
template <> inline float getMax(ushort4 &v) { return (float)std::max(std::max(v.x, v.y), std::max(v.z, v.w)); }

template <class T>
void BasicRenderer<T>::initCuda(T *h_volume, cudaExtent volumeSize, cudaExtent originalSize, float3 scale, unsigned int components, int max_error)
{
	m_scale = scale;
	m_extent = volumeSize;
	m_realExtent = originalSize;
	m_components = components;

	m_maximum = 0.0f;
	for (size_t i = 0; i < volumeSize.width * volumeSize.height * volumeSize.depth; i++) m_maximum = std::max(m_maximum, getMax(h_volume[i]));
	m_maximum = expf(logf(2.0f) * ceilf(logf(m_maximum - 1.0f) / logf(2.0f))) - 1.0f;

	// create 3D array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
	checkCudaErrors(cudaMalloc3DArray(&d_volumeArray, &channelDesc, originalSize));

	// copy data to 3D array
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr   = make_cudaPitchedPtr(h_volume, volumeSize.width*sizeof(T), volumeSize.width, volumeSize.height);
	copyParams.dstArray = d_volumeArray;
	copyParams.extent   = originalSize;
	copyParams.kind     = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParams));

	// create texture object
	{
		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = d_volumeArray;

		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.readMode = cudaReadModeNormalizedFloat;
		texDesc.addressMode[0] = cudaAddressModeClamp;
		texDesc.addressMode[1] = cudaAddressModeClamp;
		texDesc.addressMode[2] = cudaAddressModeClamp;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.normalizedCoords = true;

		// create texture object: we only have to do this once!
		checkCudaErrors(cudaCreateTextureObject(&m_volumeTex, &resDesc, &texDesc, NULL));
	}


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
		checkCudaErrors(cudaCreateTextureObject(&m_transferTex, &resDesc, &texDesc, NULL));
	}
}

template <class T>
void BasicRenderer<T>::updateTF()
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
void BasicRenderer<T>::freeCudaBuffers()
{
	if (m_transferTex != 0) checkCudaErrors(cudaDestroyTextureObject(m_transferTex));
	if (m_volumeTex != 0) checkCudaErrors(cudaDestroyTextureObject(m_volumeTex));
	if (d_volumeArray != 0) checkCudaErrors(cudaFreeArray(d_volumeArray));
	if (d_transferFuncArray != 0) checkCudaErrors(cudaFreeArray(d_transferFuncArray));
	m_transferTex = 0;
	m_volumeTex = 0;
	d_volumeArray = 0;
	d_transferFuncArray = 0;
}

template <class T>
void BasicRenderer<T>::render_kernel(dim3 gridSize, dim3 blockSize, dim3 warpDim, uint *d_output, uint imageW, uint imageH,
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

template <class T>
void BasicRenderer<T>::copyViewMatrix(float *viewMatrix, size_t sizeofMatrix)
{
	this->viewMatrix = viewMatrix;
	this->sizeofMatrix = sizeofMatrix;
}

template <class T>
void BasicRenderer<T>::copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix)
{
	this->invViewMatrix = invViewMatrix;
	this->sizeofMatrix = sizeofMatrix;
}

template <class T>
void BasicRenderer<T>::getLaunchParameter(dim3& warpDim, uint imageW, uint imageH, float tstep)
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
		p[i].x *= (float) m_realExtent.width  / m_scale.x;
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

	sampling_dist[1] *= 0.5f;
	while (warpDim.x * warpDim.y * warpDim.z < 32)
	{
		float3 d;
		d.x = warpDim.x * sampling_dist[0];
		d.y = warpDim.y * sampling_dist[1];
		d.z = warpDim.z * sampling_dist[2];
		//if ((d.x <= d.y) && ((d.x <= 1.0f) || (d.x <= d.z))) warpDim.x <<= 1;
		//else if ((d.y <= 1.0f) || (d.y <= d.z)) warpDim.y <<= 1;
		if ((d.x <= d.y) && (d.x <= d.z)) warpDim.x <<= 1;
		else if (d.y <= d.z) warpDim.y <<= 1;
		else warpDim.z <<= 1;
	}
}

// valid template types
template class BasicRenderer<unsigned char>;
template class BasicRenderer<unsigned short>;
template class BasicRenderer<uchar4>;
template class BasicRenderer<ushort4>;
