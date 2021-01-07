// Copyright (c) 2016 Stefan Guthe / GCC / TU-Darmstadt. All rights reserved. 
// Use of this source code is governed by the BSD 3-Clause license that can be
// found in the LICENSE file.

#pragma once
#include "global_defines.h"

#include "Renderer.h"
#include "cuda.h"
#include <algorithm>

template <class T>
class RendererRBUC8x8 : public Renderer<T>
{
private:
	cudaArray *d_transferFuncArray;
	bool m_bLinearFilter;
	cudaExtent m_extent;
	cudaExtent m_realExtent;
	float3 m_scale;
	bool m_rgba;
	bool m_lighting;
	float m_maximum;

	bool volume64;
	// ugly but working
	void *offsetVolume;
	void *compressedVolume;

	cudaTextureObject_t m_transferTex;

	float *invViewMatrix;
	float *viewMatrix;
	size_t sizeofMatrix;
	unsigned int m_components;
public:
	RendererRBUC8x8()
	{
		d_transferFuncArray = 0;
		m_bLinearFilter = false;
		m_rgba = false;
		m_lighting = false;
		m_components = 0;
		volume64 = false;
		offsetVolume = NULL;
		compressedVolume = NULL;
	}
	virtual ~RendererRBUC8x8();

	virtual void setTextureFilterMode(bool bLinearFilter) override ;
	virtual void initCuda(T *h_volume, cudaExtent volumeSize, cudaExtent originalSize, float3 scale, unsigned int components, int max_error) override;
	virtual void updateTF() override;
	virtual void freeCudaBuffers() override;
	virtual void render_kernel(dim3 gridSize, dim3 blockSize, dim3 warpDim, uint *d_output, uint imageW, uint imageH,
							   float density, float brightness, float transferOffset, float transferScale, float tstep, bool white) override ;
	virtual void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix) override ;
	virtual void copyViewMatrix(float *viewMatrix, size_t sizeofMatrix) override;

	virtual void enableRGBA(bool rgba) override { m_rgba = rgba; }

	virtual void enableLighting(bool lighting) override { m_lighting = lighting; }

	virtual void getLaunchParameter(dim3& warpDim, uint imageW, uint imageH, float tstep) override ;

	virtual void setScale(float3 scale) override { m_scale = scale; }

private:
	unsigned int compress(T *raw, unsigned char *comp);
	unsigned int setMinMax(unsigned char *comp, int min, int max);

private:
	template <uint wsx, uint wsy, uint wsz>
	void render_internal_kernel(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH,
		float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
};
