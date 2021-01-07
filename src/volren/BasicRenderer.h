// Copyright (c) 2016 Stefan Guthe / GCC / TU-Darmstadt. All rights reserved. 
// Use of this source code is governed by the BSD 3-Clause license that can be
// found in the LICENSE file.

#pragma once
#include "global_defines.h"

#include "Renderer.h"

template <class T>
class BasicRenderer : public Renderer<T>
{
private:
	cudaArray *d_volumeArray;
	cudaArray *d_transferFuncArray;
	cudaExtent m_extent;
	cudaExtent m_realExtent;
	float3 m_scale;
	bool m_rgba;
	bool m_lighting;
	bool m_bLinearFilter;
	float m_maximum;

	cudaTextureObject_t m_transferTex;
	cudaTextureObject_t m_volumeTex;

	float *invViewMatrix;
	float *viewMatrix;
	size_t sizeofMatrix;
	unsigned int m_components;
public:
	BasicRenderer()
	{
		d_volumeArray = 0;
		d_transferFuncArray = 0;
		m_rgba = false;
		m_lighting = false;
		m_bLinearFilter = false;
		m_transferTex = 0;
		m_volumeTex = 0;
		m_components = 0;
	}
	virtual ~BasicRenderer() { freeCudaBuffers(); } 

	virtual void setTextureFilterMode(bool bLinearFilter) override ;
	virtual void initCuda(T *h_volume, cudaExtent volumeSize, cudaExtent originalSize, float3 scale, unsigned int components, int max_error) override;
	virtual void updateTF() override;
	virtual void freeCudaBuffers() override ;
	virtual void render_kernel(dim3 gridSize, dim3 blockSize, dim3 warpDim, uint *d_output, uint imageW, uint imageH,
							   float density, float brightness, float transferOffset, float transferScale, float tstep, bool white) override ;
	virtual void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix) override;
	virtual void copyViewMatrix(float *viewMatrix, size_t sizeofMatrix) override;

	virtual void enableRGBA(bool rgba) override { m_rgba = rgba; }
	virtual void enableLighting(bool lighting) override { m_lighting = lighting; }

	virtual void getLaunchParameter(dim3& warpDim, uint imageW, uint imageH, float tstep) override;

	virtual void setScale(float3 scale) override { m_scale = scale; }

private:
	template <uint wsx, uint wsy, uint wsz>
	void render_internal_kernel(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH,
		float density, float brightness, float transferOffset, float transferScale, float tstep, bool white);
};
