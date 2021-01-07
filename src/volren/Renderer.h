// Copyright (c) 2016 Stefan Guthe / GCC / TU-Darmstadt. All rights reserved. 
// Use of this source code is governed by the BSD 3-Clause license that can be
// found in the LICENSE file.

#pragma once

#include <helper_cuda.h>
#include <helper_math.h>

typedef unsigned int  uint;
typedef unsigned char uchar;

extern float4 *transferFunc;
extern size_t transferFunc_size;

template <class T>
class Renderer
{
public:
	Renderer() {}
	virtual ~Renderer() {}

	virtual void setTextureFilterMode(bool bLinearFilter) = 0;
	virtual void initCuda(T *h_volume, cudaExtent volumeSize, cudaExtent originalSize, float3 scale, unsigned int components, int max_error) = 0;
	virtual void updateTF() = 0;
	virtual void freeCudaBuffers() = 0;
	virtual void render_kernel(dim3 gridSize, dim3 blockSize, dim3 warpDim, uint *d_output, uint imageW, uint imageH,
							   float density, float brightness, float transferOffset, float transferScale, float tstep, bool white) = 0;
	virtual void copyViewMatrix(float *viewMatrix, size_t sizeofMatrix) = 0;
	virtual void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix) = 0;
	virtual void enableRGBA(bool rgba) = 0;
	virtual void enableLighting(bool lighting) = 0;

	virtual void getLaunchParameter(dim3& warpDim, uint imageW, uint imageH, float tstep) = 0;
	virtual void setScale(float3 scale) = 0;
};
