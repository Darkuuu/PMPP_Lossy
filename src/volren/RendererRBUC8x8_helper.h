// Copyright (c) 2016 Stefan Guthe / GCC / TU-Darmstadt. All rights reserved. 
// Use of this source code is governed by the BSD 3-Clause license that can be
// found in the LICENSE file.

#pragma once

#include "CompressRBUC.h"

__forceinline__ __host__ __device__ void operator/=(uchar4 &a, int b)
{
	a.x /= b; a.y /= b; a.z /= b; a.w /= b;
}

__forceinline__ __host__ __device__ bool operator==(uchar4 a, uchar4 b)
{
	return (a.x == b.x) && (a.y == b.y) && (a.z == b.z) && (a.w == b.w);
}

__forceinline__ __host__ __device__ bool operator==(uchar4 a, uchar b)
{
	return (a.x == b) && (a.y == b) && (a.z == b) && (a.w == b);
}

__forceinline__ __host__ __device__ uchar4 operator/(uchar4 a, uchar b)
{
	return make_uchar4(a.x / b, a.y / b, a.z / b, a.w / b);
}

__forceinline__ __host__ __device__ bool operator!=(uchar4 a, uchar4 b)
{
	return (a.x != b.x) || (a.y != b.y) || (a.z != b.z) || (a.w != b.w);
}

__forceinline__ __host__ __device__ void operator+=(uchar4 &a, uchar4 b)
{
	a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}

__forceinline__ __host__ __device__ void operator-=(uchar4 &a, uchar4 b)
{
	a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}

__forceinline__ __host__ __device__ void operator+=(int4 &a, uchar4 b)
{
	a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}

__forceinline__ __host__ __device__ void operator-=(int4 &a, uchar4 b)
{
	a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}

__forceinline__ __host__ __device__ uchar4 operator-(uchar4 a, uchar4 b)
{
	return make_uchar4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

__forceinline__ __host__ __device__ void operator/=(ushort4 &a, int b)
{
	a.x /= b; a.y /= b; a.z /= b; a.w /= b;
}

__forceinline__ __host__ __device__ bool operator==(ushort4 a, ushort4 b)
{
	return (a.x == b.x) && (a.y == b.y) && (a.z == b.z) && (a.w == b.w);
}

__forceinline__ __host__ __device__ bool operator==(ushort4 a, ushort b)
{
	return (a.x == b) && (a.y == b) && (a.z == b) && (a.w == b);
}

__forceinline__ __host__ __device__ ushort4 operator/(ushort4 a, ushort b)
{
	return make_ushort4(a.x / b, a.y / b, a.z / b, a.w / b);
}

__forceinline__ __host__ __device__ bool operator!=(ushort4 a, ushort4 b)
{
	return (a.x != b.x) || (a.y != b.y) || (a.z != b.z) || (a.w != b.w);
}

__forceinline__ __host__ __device__ void operator+=(ushort4 &a, ushort4 b)
{
	a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}

__forceinline__ __host__ __device__ void operator-=(ushort4 &a, ushort4 b)
{
	a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}

__forceinline__ __host__ __device__ void operator+=(int4 &a, ushort4 b)
{
	a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}

__forceinline__ __host__ __device__ void operator-=(int4 &a, ushort4 b)
{
	a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}

__forceinline__ __host__ __device__ ushort4 operator-(ushort4 a, ushort4 b)
{
	return make_ushort4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
