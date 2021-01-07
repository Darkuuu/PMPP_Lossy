// Copyright (c) 2016 Stefan Guthe / GCC / TU-Darmstadt. All rights reserved. 
// Use of this source code is governed by the BSD 3-Clause license that can be
// found in the LICENSE file.

#pragma once

#include <cuda.h>
#include "CudaIntrinsics.h"
#include "global_defines.h"
#include "volumeRenderer_helper.h"

// fix light vector
static const float3 lDir = make_float3(0.25f, -0.25f, 0.93541434669348534639593718307914f);

template <unsigned int wsx, unsigned int wsz>
__device__ __forceinline__ uint getX()
{
	uint widx;
	switch (wsz)
	{
	case 1:
		widx = threadIdx.x;
		break;
	case 2:
		widx = threadIdx.x >> 1;
		break;
	case 4:
		widx = threadIdx.x >> 2;
		break;
	case 8:
		widx = threadIdx.x >> 3;
		break;
	case 16:
		widx = threadIdx.x >> 4;
		break;
	default:
		widx = 0;
	}
	switch (wsx)
	{
	case 1:
		return blockDim.y * blockIdx.x + threadIdx.y;
	case 2:
		return ((blockDim.y * blockIdx.x + threadIdx.y) << 1) + (widx & 1);
	case 4:
		return ((blockDim.y * blockIdx.x + threadIdx.y) << 2) + (widx & 3);
	case 8:
		return ((blockDim.y * blockIdx.x + threadIdx.y) << 3) + (widx & 7);
	case 16:
		return ((blockDim.y * blockIdx.x + threadIdx.y) << 4) + (widx & 15);
	default:
		return ((blockDim.y * blockIdx.x + threadIdx.y) << 5) + widx;
	}
}

template <unsigned int wsx, unsigned int wsy, unsigned int wsz>
__device__ __forceinline__ uint getY()
{
	switch (wsy)
	{
	case 1:
		return blockDim.z * blockIdx.y + threadIdx.z;
	case 2:
		return ((blockDim.z * blockIdx.y + threadIdx.z) << 1) + (threadIdx.x >> 4);
	case 4:
		return ((blockDim.z * blockIdx.y + threadIdx.z) << 2) + (threadIdx.x >> 3);
	case 8:
		return ((blockDim.z * blockIdx.y + threadIdx.z) << 3) + (threadIdx.x >> 2);
	case 16:
		return ((blockDim.z * blockIdx.y + threadIdx.z) << 4) + (threadIdx.x >> 1);
	default:
		return ((blockDim.z * blockIdx.y + threadIdx.z) << 5) + threadIdx.x;
	}
}

template <unsigned int wsz>
__device__ __forceinline__ bool checkBounds(const uint x, const uint y, const uint imageW, const uint imageH)
{
	return __all((x >= imageW) || (y >= imageH));
}

template <>
__device__ __forceinline__ bool checkBounds<32>(const uint x, const uint y, const uint imageW, const uint imageH)
{
	return ((x >= imageW) || (y >= imageH));
}

template <unsigned int wsz>
__device__ __forceinline__ bool getEnable(const uint x, const uint y, const uint imageW, const uint imageH)
{
	return ((x < imageW) && (y < imageH));
}

template <>
__device__ __forceinline__ bool getEnable<32>(const uint x, const uint y, const uint imageW, const uint imageH)
{
	return true;
}

template <unsigned int wsz>
__device__ __forceinline__ bool getAny(const bool b)
{
	return __any(b);
}

template <>
__device__ __forceinline__ bool getAny<32>(const bool b)
{
	return b;
}

template <unsigned int wsz>
__device__ __forceinline__ void calcOver(float4 &sum, float4 &col, const uint lane);

template<>
__device__ __forceinline__ void calcOver<1>(float4 &sum, float4 &col, const uint lane)
{
	sum = sum + col*(1.0f - sum.w);
}

template<>
__device__ __forceinline__ void calcOver<2>(float4 &sum, float4 &col, const uint lane)
{
	float4 col2;
	col2.x = __shfl_xor(col.x, 1);
	col2.y = __shfl_xor(col.y, 1);
	col2.z = __shfl_xor(col.z, 1);
	col2.w = __shfl_xor(col.w, 1);
	col = col + col2*(1.0f - col.w);
	col.x = __shfl(col.x, lane & 0x1e);
	col.y = __shfl(col.y, lane & 0x1e);
	col.z = __shfl(col.z, lane & 0x1e);
	col.w = __shfl(col.w, lane & 0x1e);
	sum = sum + col*(1.0f - sum.w);
}

template<>
__device__ __forceinline__ void calcOver<4>(float4 &sum, float4 &col, const uint lane)
{
	float4 col2;
	col2.x = __shfl_xor(col.x, 1);
	col2.y = __shfl_xor(col.y, 1);
	col2.z = __shfl_xor(col.z, 1);
	col2.w = __shfl_xor(col.w, 1);
	col = col + col2*(1.0f - col.w);
	col2.x = __shfl_xor(col.x, 2);
	col2.y = __shfl_xor(col.y, 2);
	col2.z = __shfl_xor(col.z, 2);
	col2.w = __shfl_xor(col.w, 2);
	col = col + col2*(1.0f - col.w);
	col.x = __shfl(col.x, lane & 0x1c);
	col.y = __shfl(col.y, lane & 0x1c);
	col.z = __shfl(col.z, lane & 0x1c);
	col.w = __shfl(col.w, lane & 0x1c);
	sum = sum + col*(1.0f - sum.w);
}

template<>
__device__ __forceinline__ void calcOver<8>(float4 &sum, float4 &col, const uint lane)
{
	float4 col2;
	col2.x = __shfl_xor(col.x, 1);
	col2.y = __shfl_xor(col.y, 1);
	col2.z = __shfl_xor(col.z, 1);
	col2.w = __shfl_xor(col.w, 1);
	col = col + col2*(1.0f - col.w);
	col2.x = __shfl_xor(col.x, 2);
	col2.y = __shfl_xor(col.y, 2);
	col2.z = __shfl_xor(col.z, 2);
	col2.w = __shfl_xor(col.w, 2);
	col = col + col2*(1.0f - col.w);
	col2.x = __shfl_xor(col.x, 4);
	col2.y = __shfl_xor(col.y, 4);
	col2.z = __shfl_xor(col.z, 4);
	col2.w = __shfl_xor(col.w, 4);
	col = col + col2*(1.0f - col.w);
	col.x = __shfl(col.x, lane & 0x18);
	col.y = __shfl(col.y, lane & 0x18);
	col.z = __shfl(col.z, lane & 0x18);
	col.w = __shfl(col.w, lane & 0x18);
	sum = sum + col*(1.0f - sum.w);
}

template<>
__device__ __forceinline__ void calcOver<16>(float4 &sum, float4 &col, const uint lane)
{
	float4 col2;
	col2.x = __shfl_xor(col.x, 1);
	col2.y = __shfl_xor(col.y, 1);
	col2.z = __shfl_xor(col.z, 1);
	col2.w = __shfl_xor(col.w, 1);
	col = col + col2*(1.0f - col.w);
	col2.x = __shfl_xor(col.x, 2);
	col2.y = __shfl_xor(col.y, 2);
	col2.z = __shfl_xor(col.z, 2);
	col2.w = __shfl_xor(col.w, 2);
	col = col + col2*(1.0f - col.w);
	col2.x = __shfl_xor(col.x, 4);
	col2.y = __shfl_xor(col.y, 4);
	col2.z = __shfl_xor(col.z, 4);
	col2.w = __shfl_xor(col.w, 4);
	col = col + col2*(1.0f - col.w);
	col2.x = __shfl_xor(col.x, 8);
	col2.y = __shfl_xor(col.y, 8);
	col2.z = __shfl_xor(col.z, 8);
	col2.w = __shfl_xor(col.w, 8);
	col = col + col2*(1.0f - col.w);
	col.x = __shfl(col.x, lane & 0x10);
	col.y = __shfl(col.y, lane & 0x10);
	col.z = __shfl(col.z, lane & 0x10);
	col.w = __shfl(col.w, lane & 0x10);
	sum = sum + col*(1.0f - sum.w);
}

template<>
__device__ __forceinline__ void calcOver<32>(float4 &sum, float4 &col, const uint lane)
{
	float4 col2;
	col2.x = __shfl_xor(col.x, 1);
	col2.y = __shfl_xor(col.y, 1);
	col2.z = __shfl_xor(col.z, 1);
	col2.w = __shfl_xor(col.w, 1);
	col = col + col2*(1.0f - col.w);
	col2.x = __shfl_xor(col.x, 2);
	col2.y = __shfl_xor(col.y, 2);
	col2.z = __shfl_xor(col.z, 2);
	col2.w = __shfl_xor(col.w, 2);
	col = col + col2*(1.0f - col.w);
	col2.x = __shfl_xor(col.x, 4);
	col2.y = __shfl_xor(col.y, 4);
	col2.z = __shfl_xor(col.z, 4);
	col2.w = __shfl_xor(col.w, 4);
	col = col + col2*(1.0f - col.w);
	col2.x = __shfl_xor(col.x, 8);
	col2.y = __shfl_xor(col.y, 8);
	col2.z = __shfl_xor(col.z, 8);
	col2.w = __shfl_xor(col.w, 8);
	col = col + col2*(1.0f - col.w);
	col2.x = __shfl_xor(col.x, 16);
	col2.y = __shfl_xor(col.y, 16);
	col2.z = __shfl_xor(col.z, 16);
	col2.w = __shfl_xor(col.w, 16);
	col = col + col2*(1.0f - col.w);
	col.x = __shfl(col.x, 0);
	col.y = __shfl(col.y, 0);
	col.z = __shfl(col.z, 0);
	col.w = __shfl(col.w, 0);
	sum = sum + col*(1.0f - sum.w);
}

template <unsigned int wsx, unsigned int wsy, unsigned int wsz, class S, class T>
__global__ void
CudaRenderer(S sampler, T transferFunction, uint *d_output, uint imageW, uint imageH,
float density, float brightness,
float transferOffset, float transferScale, float sampleScale, float3 scale, float tstep, bool white)
{
	sampler.init();

	const int maxSteps = (int)(5.0f / (tstep * wsz));
	const float opacityThreshold = 0.995f;
	const float3 boxMin = make_float3(-scale.x, -scale.y, -scale.z);
	const float3 &boxMax = scale;

	// reorganize threads so every warp covers 8x4 pixel
	const uint x = getX<wsx, wsz>();
	const uint y = getY<wsx, wsy, wsz>();

	const uint lane = __laneid();

	if (checkBounds<wsz>(x, y, imageW, imageH)) return;

	bool enabled = getEnable<wsz>(x, y, imageW, imageH);

	float u = (x / (float)imageW)*2.0f - 1.0f;
	float v = (y / (float)imageH)*2.0f - 1.0f;

	// calculate eye ray in world space
	Ray eyeRay;
	eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
	eyeRay.d = normalize(make_float3(u, v, -2.0f));
	eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

	// find intersection with box
	float tnear, tfar;
	bool active = enabled && intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);
	if (tnear < 0.5f * tstep) tnear = 0.5f * tstep;     // clamp to near plane
	if (tnear > tfar) active = false;					// check if we are still in the volume

	// march along ray from front to back, accumulating color
	float4 sum = make_float4(0.0f);

	if (getAny<wsz>(active))
	{
		float t = tnear;
		float3 pos = eyeRay.o + eyeRay.d*t;
		float3 step = eyeRay.d*tstep;

		// scale position and step to be in [0,1] range
		pos.x += scale.x;
		pos.y += scale.y;
		pos.z += scale.z;
		pos.x /= 2.0f * scale.x;
		pos.y /= 2.0f * scale.y;
		pos.z /= 2.0f * scale.z;
		step.x /= 2.0f * scale.x;
		step.y /= 2.0f * scale.y;
		step.z /= 2.0f * scale.z;

		// each thread gets a different sample position
		switch (wsz)
		{
		case 1:
			break;
		case 2:
			if ((lane & 1) != 0)
			{
				pos.x += step.x;
				pos.y += step.y;
				pos.z += step.z;
				t += tstep;
			}
			step.x += step.x;
			step.y += step.y;
			step.z += step.z;
			tstep += tstep;
			if (t > tfar) active = false;
			break;
		case 4:
			{
				const float sample_idx = (float)(lane & 3);
				pos.x += step.x * sample_idx;
				pos.y += step.y * sample_idx;
				pos.z += step.z * sample_idx;
				t += tstep * sample_idx;
				step.x *= 4.0f;
				step.y *= 4.0f;
				step.z *= 4.0f;
				tstep *= 4.0f;
				if (t > tfar) active = false;
			}
			break;
		case 8:
			{
				const float sample_idx = (float)(lane & 7);
				pos.x += step.x * sample_idx;
				pos.y += step.y * sample_idx;
				pos.z += step.z * sample_idx;
				t += tstep * sample_idx;
				step.x *= 8.0f;
				step.y *= 8.0f;
				step.z *= 8.0f;
				tstep *= 8.0f;
				if (t > tfar) active = false;
			}
			break;
		case 16:
			{
				const float sample_idx = (float)(lane & 15);
				pos.x += step.x * sample_idx;
				pos.y += step.y * sample_idx;
				pos.z += step.z * sample_idx;
				t += tstep * sample_idx;
				step.x *= 16.0f;
				step.y *= 16.0f;
				step.z *= 16.0f;
				tstep *= 16.0f;
				if (t > tfar) active = false;
			}
			break;
		default:
			{
				const float sample_idx = (float)lane;
				pos.x += step.x * sample_idx;
				pos.y += step.y * sample_idx;
				pos.z += step.z * sample_idx;
				t += tstep * sample_idx;
				step.x *= 32.0f;
				step.y *= 32.0f;
				step.z *= 32.0f;
				tstep *= 32.0f;
				if (t > tfar) active = false;
			}
		}

		for (int i = 0; i < maxSteps; i++)
		{
			// read from 3D volume
			auto sample = sampler.sample(active && (t >= tnear), pos) * sampleScale;

			float4 col = transferFunction.tf(active && (t >= tnear), sample, transferOffset, transferScale, density);

			// "over" operator for front-to-back blending
			calcOver<wsz>(sum, col, lane);

			t += tstep;

			// exit early if opaque
			if ((sum.w > opacityThreshold) || (t > tfar)) active = false;

			if (!getAny<wsz>(active)) break;

			pos += step;
		}

		sum.x *= brightness;
		sum.y *= brightness;
		sum.z *= brightness;
	}

	if (white)
	{
		sum.x += (1.0f - sum.w);
		sum.y += (1.0f - sum.w);
		sum.z += (1.0f - sum.w);
	}

	switch (wsz)
	{
	case 1:
		break;
	case 2:
		enabled &= ((lane & 1) == 0);
		break;
	case 4:
		enabled &= ((lane & 3) == 0);
		break;
	case 8:
		enabled &= ((lane & 7) == 0);
		break;
	case 16:
		enabled &= ((lane & 15) == 0);
		break;
	default:
		enabled &= (lane == 0);
	}

	// write output color
	if (enabled) d_output[y*imageW + x] = rgbaFloatToInt(sum);
}

template <unsigned int wsx, unsigned int wsy, unsigned int wsz, class S, class T>
__global__ void
CudaDerivatingRenderer(S sampler, T transferFunction, uint *d_output, uint imageW, uint imageH,
float density, float brightness,
float transferOffset, float transferScale, float sampleScale, float3 scale, float tstep, bool white)
{
	sampler.init();

	const int maxSteps = (int)(5.0f / (tstep * wsz));
	const float opacityThreshold = 0.995f;
	const float3 boxMin = make_float3(-scale.x, -scale.y, -scale.z);
	const float3 &boxMax = scale;

	// reorganize threads so every warp covers 8x4 pixel
	const uint x = getX<wsx, wsz>();
	const uint y = getY<wsx, wsy, wsz>();

	const uint lane = __laneid();

	if (checkBounds<wsz>(x, y, imageW, imageH)) return;

	bool enabled = getEnable<wsz>(x, y, imageW, imageH);

	float u = (x / (float)imageW)*2.0f - 1.0f;
	float v = (y / (float)imageH)*2.0f - 1.0f;

	// calculate eye ray in world space
	Ray eyeRay;
	eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
	eyeRay.d = normalize(make_float3(u, v, -2.0f));
	eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

	// find intersection with box
	float tnear, tfar;
	bool active = enabled && intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);
	if (tnear < 0.5f * tstep) tnear = 0.5f * tstep;     // clamp to near plane
	if (tnear > tfar) active = false;					// check if we are still in the volume

	{
		active |= __shfl_xor(active, wsz);
		active |= __shfl_xor(active, wsx * wsz);
	}

	// march along ray from front to back, accumulating color
	float4 sum = make_float4(0.0f);

	float tstep_orig = tstep;

	if (getAny<wsz>(active))
	{
		float t = tnear;
		{
			float tmp = __shfl_xor(t, wsz);
			if (tmp < t) t = tmp;
			tmp = __shfl_xor(t, wsx * wsz);
			if (tmp < t) t = tmp;
		}

		float3 pos = eyeRay.o + eyeRay.d*t;
		float3 step = eyeRay.d*tstep;

		// scale position and step to be in [0,1] range
		pos.x += scale.x;
		pos.y += scale.y;
		pos.z += scale.z;
		pos.x /= 2.0f * scale.x;
		pos.y /= 2.0f * scale.y;
		pos.z /= 2.0f * scale.z;
		step.x /= 2.0f * scale.x;
		step.y /= 2.0f * scale.y;
		step.z /= 2.0f * scale.z;

		// each thread gets a different sample position
		switch (wsz)
		{
			case 1:
				break;
			case 2:
				if ((lane & 1) != 0)
				{
					pos.x += step.x;
					pos.y += step.y;
					pos.z += step.z;
					t += tstep;
				}
				step.x += step.x;
				step.y += step.y;
				step.z += step.z;
				tstep += tstep;
				if (t > tfar) active = false;
				break;
			case 4:
			{
				const float sample_idx = (float)(lane & 3);
				pos.x += step.x * sample_idx;
				pos.y += step.y * sample_idx;
				pos.z += step.z * sample_idx;
				t += tstep * sample_idx;
				step.x *= 4.0f;
				step.y *= 4.0f;
				step.z *= 4.0f;
				tstep *= 4.0f;
				if (t > tfar) active = false;
			}
			break;
			case 8:
			{
				const float sample_idx = (float)(lane & 7);
				pos.x += step.x * sample_idx;
				pos.y += step.y * sample_idx;
				pos.z += step.z * sample_idx;
				t += tstep * sample_idx;
				step.x *= 8.0f;
				step.y *= 8.0f;
				step.z *= 8.0f;
				tstep *= 8.0f;
				if (t > tfar) active = false;
			}
			break;
			case 16:
			{
				const float sample_idx = (float)(lane & 15);
				pos.x += step.x * sample_idx;
				pos.y += step.y * sample_idx;
				pos.z += step.z * sample_idx;
				t += tstep * sample_idx;
				step.x *= 16.0f;
				step.y *= 16.0f;
				step.z *= 16.0f;
				tstep *= 16.0f;
				if (t > tfar) active = false;
			}
			break;
			default:
			{
				const float sample_idx = (float)lane;
				pos.x += step.x * sample_idx;
				pos.y += step.y * sample_idx;
				pos.z += step.z * sample_idx;
				t += tstep * sample_idx;
				step.x *= 32.0f;
				step.y *= 32.0f;
				step.z *= 32.0f;
				tstep *= 32.0f;
				if (t > tfar) active = false;
			}
		}

		float old_sample = 0.0f;
		for (int i = 0; i < maxSteps; i++)
		{
			float4 sample;
			// we still need our neighbors
			if (wsz != 1) active |= __shfl_xor(active, 1);
			active |= __shfl_xor(active, wsz);
			active |= __shfl_xor(active, wsx * wsz);
			sample.w = sampler.sample(active, pos) * sampleScale;
			float3 dir0, dir1, dir2;
			float3 a;

			a.x = sample.w - __shfl_xor(sample.w, wsz);
			dir0.x = pos.x - __shfl_xor(pos.x, wsz);
			dir0.y = pos.y - __shfl_xor(pos.y, wsz);
			dir0.z = pos.z - __shfl_xor(pos.z, wsz);
			a.y = sample.w - __shfl_xor(sample.w, wsx * wsz);
			dir1.x = pos.x - __shfl_xor(pos.x, wsx * wsz);
			dir1.y = pos.y - __shfl_xor(pos.y, wsx * wsz);
			dir1.z = pos.z - __shfl_xor(pos.z, wsx * wsz);
			if (wsz == 1)
			{
				a.z = sample.w - old_sample;
				dir2 = step;
			}
			else
			{
				a.z = sample.w - __shfl_xor(sample.w, 1);
				dir2.x = pos.x - __shfl_xor(pos.x, 1);
				dir2.y = pos.y - __shfl_xor(pos.y, 1);
				dir2.z = pos.z - __shfl_xor(pos.z, 1);
			}
			// normalize
			{
				float d = 1.0f / sqrtf(dir0.x * dir0.x + dir0.y * dir0.y + dir0.z * dir0.z);
				dir0.x *= d;
				dir0.y *= d;
				dir0.z *= d;
				a.x *= d;
			}
			{
				float d = 1.0f / sqrtf(dir1.x * dir1.x + dir1.y * dir1.y + dir1.z * dir1.z);
				dir1.x *= d;
				dir1.y *= d;
				dir1.z *= d;
				a.y *= d;
			}
			{
				float d = 1.0f / sqrtf(dir2.x * dir2.x + dir2.y * dir2.y + dir2.z * dir2.z);
				dir2.x *= d;
				dir2.y *= d;
				dir2.z *= d;
				a.z *= d;
			}
			float det =
				(dir0.x * dir1.y * dir2.z + dir0.y * dir1.z * dir2.x + dir0.z * dir1.x * dir2.y) -
				(dir0.x * dir1.z * dir2.y + dir0.y * dir1.x * dir2.z + dir0.z * dir1.y * dir2.x);
			det = tstep_orig / det;
			sample.x = det * (((dir1.y * dir2.z) - (dir1.z * dir2.y)) * a.x +
							  ((dir1.z * dir2.x) - (dir1.x * dir2.z)) * a.y +
							  ((dir1.x * dir2.y) - (dir1.y * dir2.x)) * a.z);
			sample.y = det * (((dir2.y * dir0.z) - (dir2.z * dir0.y)) * a.x +
							  ((dir2.z * dir0.x) - (dir2.x * dir0.z)) * a.y +
							  ((dir2.x * dir0.y) - (dir2.y * dir0.x)) * a.z);
			sample.z = det * (((dir0.y * dir1.z) - (dir0.z * dir1.y)) * a.x +
							  ((dir0.z * dir1.x) - (dir0.x * dir1.z)) * a.y +
							  ((dir0.x * dir1.y) - (dir0.y * dir1.x)) * a.z);

			sample.x = 0.5f * sample.x + 0.5f;
			sample.y = 0.5f * sample.y + 0.5f;
			sample.z = 0.5f * sample.z + 0.5f;

			if (wsz == 1)
			{
				old_sample = sample.w;
			}

			// only evaluate this inside the volume
			float4 col = transferFunction.tf(active && (t >= tnear) && (t <= tfar), sample, transferOffset, transferScale, density);

			// "over" operator for front-to-back blending
			calcOver<wsz>(sum, col, lane);

			t += tstep;

			// exit early if opaque
			if ((sum.w > opacityThreshold) || (t > tfar)) active = false;

			if (!getAny<wsz>(active)) break;

			pos += step;
		}

		sum.x *= brightness;
		sum.y *= brightness;
		sum.z *= brightness;
	}

	if (white)
	{
		sum.x += (1.0f - sum.w);
		sum.y += (1.0f - sum.w);
		sum.z += (1.0f - sum.w);
	}

	switch (wsz)
	{
	case 1:
		break;
	case 2:
		enabled &= ((lane & 1) == 0);
		break;
	case 4:
		enabled &= ((lane & 3) == 0);
		break;
	case 8:
		enabled &= ((lane & 7) == 0);
		break;
	case 16:
		enabled &= ((lane & 15) == 0);
		break;
	default:
		enabled &= (lane == 0);
	}

	// write output color
	if (enabled) d_output[y*imageW + x] = rgbaFloatToInt(sum);
}

