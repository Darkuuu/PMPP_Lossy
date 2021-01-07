// Copyright (c) 2016 Stefan Guthe / GCC / TU-Darmstadt. All rights reserved. 
// Use of this source code is governed by the BSD 3-Clause license that can be
// found in the LICENSE file.

#pragma once

#include "volumeRenderer_helper.h"

class SimpleTransferFunction
{
private:
	cudaTextureObject_t transferTex;
public:
	SimpleTransferFunction(cudaTextureObject_t transferTex)
	{
		this->transferTex = transferTex;
	}
	__device__ __forceinline__ float4 tf(bool active, float scalar, float transferOffset, float transferScale, float density)
	{
		if (!active) return make_float4(0.0f, 0.0f, 0.0f, 0.0f);

		float4 col = tex1D<float4>(transferTex, (scalar - transferOffset)*transferScale);
		col.w = __saturatef(col.w * density);

		// pre-multiply alpha
		col.x *= col.w;
		col.y *= col.w;
		col.z *= col.w;
		return col;
	}
};

class DiscardColorTransferFunction
{
private:
	cudaTextureObject_t transferTex;
public:
	DiscardColorTransferFunction(cudaTextureObject_t transferTex)
	{
		this->transferTex = transferTex;
	}
	__device__ __forceinline__ float4 tf(bool active, float4 scalar, float transferOffset, float transferScale, float density)
	{
		if (!active) return make_float4(0.0f, 0.0f, 0.0f, 0.0f);

		float4 col = tex1D<float4>(transferTex, (scalar.w - transferOffset)*transferScale);
		col.w = __saturatef(col.w * density);

		// pre-multiply alpha
		col.x *= col.w;
		col.y *= col.w;
		col.z *= col.w;
		return col;
	}
};

class ColorTransferFunction
{
public:
	__device__ __forceinline__ float4 tf(bool active, float4 col, float transferOffset, float transferScale, float density)
	{
		if (!active) return make_float4(0.0f, 0.0f, 0.0f, 0.0f);

		col.w = __saturatef((col.w - transferOffset)*transferScale);
		col.w = __saturatef(col.w * density);

		// pre-multiply alpha
		col.x *= col.w;
		col.y *= col.w;
		col.z *= col.w;
		return col;
	}
};

class LightingTransferFunction
{
private:
	cudaTextureObject_t transferTex;
public:
	LightingTransferFunction(cudaTextureObject_t transferTex)
	{
		this->transferTex = transferTex;
	}
	__device__ __forceinline__ float4 tf(bool active, float4 sample, float transferOffset, float transferScale, float density)
	{
		if (!active) return make_float4(0.0f, 0.0f, 0.0f, 0.0f);

		float4 col = tex1D<float4>(transferTex, (sample.w - transferOffset)*transferScale);
		sample.x = 2.0f * sample.x - 1.0f;
		sample.y = 2.0f * sample.y - 1.0f;
		sample.z = 2.0f * sample.z - 1.0f;
		float l = sample.x * sample.x + sample.y * sample.y + sample.z * sample.z;
		if (l == 0.0f)
		{
			col.x *= 0.25f;
			col.y *= 0.25f;
			col.z *= 0.25f;
		}
		else
		{
			l = __frsqrt_rn(l);
			l = __saturatef(l * (sample.x * c_lDir.x + sample.y * c_lDir.y + sample.z * c_lDir.z) + 0.25f);
			col.x *= l;
			col.y *= l;
			col.z *= l;
		}
		col.w = __saturatef(col.w * density);

		// pre-multiply alpha
		col.x *= col.w;
		col.y *= col.w;
		col.z *= col.w;

		return col;
	}
};

class LightingGradientModulationTransferFunction
{
private:
	cudaTextureObject_t transferTex;
public:
	LightingGradientModulationTransferFunction(cudaTextureObject_t transferTex)
	{
		this->transferTex = transferTex;
	}
	__device__ __forceinline__ float4 tf(bool active, float4 sample, float transferOffset, float transferScale, float density)
	{
		if (!active) return make_float4(0.0f, 0.0f, 0.0f, 0.0f);

		float4 col = tex1D<float4>(transferTex, (sample.w - transferOffset)*transferScale);
		sample.x = 2.0f * sample.x - 1.0f;
		sample.y = 2.0f * sample.y - 1.0f;
		sample.z = 2.0f * sample.z - 1.0f;
		float l = sample.x * sample.x + sample.y * sample.y + sample.z * sample.z;
		if (l == 0.0f)
			col.x = col.y = col.z = col.w = 0.0f;
		else
		{
			col.w *= l;
			l = __frsqrt_rn(l);
			l = __saturatef(l * (sample.x * c_lDir.x + sample.y * c_lDir.y + sample.z * c_lDir.z) + 0.25f);
			col.x *= l;
			col.y *= l;
			col.z *= l;
		}
		col.w = __saturatef(col.w * density);

		// pre-multiply alpha
		col.x *= col.w;
		col.y *= col.w;
		col.z *= col.w;

		return col;
	}
};
