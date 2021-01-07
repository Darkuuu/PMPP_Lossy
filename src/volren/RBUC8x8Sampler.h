// Copyright (c) 2016 Stefan Guthe / GCC / TU-Darmstadt. All rights reserved. 
// Use of this source code is governed by the BSD 3-Clause license that can be
// found in the LICENSE file.

#pragma once

#include "global_defines.h"
#include "CudaRenderer.h"
#include "OffsetVolume.h"
#include "CompressedVolume.h"
#include "CachedVolume.h"

template <typename CV, typename AT, typename DT, typename USAGE, typename SAMPLE>
class RBUC8x8Sampler
{
protected:
	// constructor
	CV volume;
	AT addressOffset;
private:
	RBUC8x8Sampler() {}
public:
	RBUC8x8Sampler(const CV &volume, AT addressOffset) : volume(volume), addressOffset(addressOffset) {}
	RBUC8x8Sampler(const RBUC8x8Sampler &a) : volume(a.volume), addressOffset(a.addressOffset) {}
	~RBUC8x8Sampler() {}

	__device__ __forceinline__ void init()
	{
		volume.init((AT*)&(array[addressOffset]), (DT)array);
	}
	__device__ __forceinline__ SAMPLE sample(bool active, float3 pos)
	{
		pos.x = __saturatef(pos.x);
		pos.y = __saturatef(pos.y);
		pos.z = __saturatef(pos.z);

		USAGE usage;
		usage.init();
		return volume.sample(active, pos);
	}
};

template <typename CV, typename AT, typename DT, typename USAGE, typename SAMPLE>
class RBUC8x8Sampler4
{
protected:
	// constructor
	CV volume;
	AT addressOffset;
	float comp_add, comp_scale;
public:
private:
	RBUC8x8Sampler4() {}
public:
	RBUC8x8Sampler4(const CV &volume, AT addressOffset, float comp_add, float comp_scale) : volume(volume), addressOffset(addressOffset), comp_add(comp_add), comp_scale(comp_scale) {}
	RBUC8x8Sampler4(const RBUC8x8Sampler4 &a) : volume(a.volume), addressOffset(a.addressOffset), comp_add(a.comp_add), comp_scale(a.comp_scale) {}
	~RBUC8x8Sampler4() {}

	__device__ __forceinline__ void init()
	{
		volume.init((AT*)&(array[addressOffset]), (DT)array);
	}
	__device__ __forceinline__ SAMPLE sample(bool active, float3 pos)
	{
		pos.x = __saturatef(pos.x);
		pos.y = __saturatef(pos.y);
		pos.z = __saturatef(pos.z);
		SAMPLE col;
		pos.z *= comp_scale;
		float tmp;

		USAGE usage;
		usage.init();

#pragma unroll 1
		for (unsigned int i = 0; i < 4; i++)
		{
			tmp = volume.sample(active, pos);
			switch (i) { case 0: col.x = tmp; break; case 1: col.y = tmp; break; case 2: col.z = tmp; break; default: col.w = tmp; }
			pos.z += comp_add;
			usage.reinit();
		}
		return col;
	}
};

