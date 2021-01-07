// Copyright (c) 2016 Stefan Guthe / GCC / TU-Darmstadt. All rights reserved. 
// Use of this source code is governed by the BSD 3-Clause license that can be
// found in the LICENSE file.

#pragma once

class TextureSampler8
{
private:
	cudaTextureObject_t tex;
public:
	TextureSampler8(cudaTextureObject_t tex)
	{
		this->tex = tex;
	}
	__device__ __forceinline__ void init() {}
	__device__ __forceinline__ float sample(bool active, float3 pos)
	{
		if (active)
			return tex3D<float>(tex, pos.x, pos.y, pos.z);
		else
			return 0.0f;
	}
};

class TextureSampler16
{
private:
	cudaTextureObject_t tex;
public:
	TextureSampler16(cudaTextureObject_t tex)
	{
		this->tex = tex;
	}
	__device__ __forceinline__ void init() {}
	__device__ __forceinline__ float sample(bool active, float3 pos)
	{
		if (active)
			return tex3D<float>(tex, pos.x, pos.y, pos.z);
		else
			return 0.0f;
	}
};

class TextureSampler8x4
{
private:
	float comp_add, comp_scale;
	cudaTextureObject_t tex;
public:
	TextureSampler8x4(cudaTextureObject_t tex, float comp_add, float comp_scale)
	{
		this->tex = tex;
		this->comp_add = comp_add;
		this->comp_scale = comp_scale;
	}

	__device__ __forceinline__ void init() {}
	__device__ __forceinline__ float4 sample(bool active, float3 pos)
	{
		pos.z *= comp_scale;
		if (active)
			return make_float4(
			tex3D<float>(tex, pos.x, pos.y, pos.z),
			tex3D<float>(tex, pos.x, pos.y, pos.z + comp_add),
			tex3D<float>(tex, pos.x, pos.y, pos.z + 2.0f * comp_add),
			tex3D<float>(tex, pos.x, pos.y, pos.z + 3.0f * comp_add));
		else
			return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	}
};

class TextureSampler16x4
{
private:
	float comp_add, comp_scale;
	cudaTextureObject_t tex;
public:
	TextureSampler16x4(cudaTextureObject_t tex, float comp_add, float comp_scale)
	{
		this->tex = tex;
		this->comp_add = comp_add;
		this->comp_scale = comp_scale;
	}

	__device__ __forceinline__ void init() {}
	__device__ __forceinline__ float4 sample(bool active, float3 pos)
	{
		pos.z *= comp_scale;
		if (active)
			return make_float4(
			tex3D<float>(tex, pos.x, pos.y, pos.z),
			tex3D<float>(tex, pos.x, pos.y, pos.z + comp_add),
			tex3D<float>(tex, pos.x, pos.y, pos.z + 2.0f * comp_add),
			tex3D<float>(tex, pos.x, pos.y, pos.z + 3.0f * comp_add));
		else
			return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	}
};

class TextureSampler32
{
private:
	cudaTextureObject_t tex;
public:
	TextureSampler32(cudaTextureObject_t tex)
	{
		this->tex = tex;
	}
	__device__ __forceinline__ void init() {}
	__device__ __forceinline__ float4 sample(bool active, float3 pos)
	{
		if (active)
			return tex3D<float4>(tex, pos.x, pos.y, pos.z);
		else
			return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	}
};

class TextureSampler64
{
private:
	cudaTextureObject_t tex;
public:
	TextureSampler64(cudaTextureObject_t tex)
	{
		this->tex = tex;
	}
	__device__ __forceinline__ void init() {}
	__device__ __forceinline__ float4 sample(bool active, float3 pos)
	{
		if (active)
			return tex3D<float4>(tex, pos.x, pos.y, pos.z);
		else
			return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	}
};

