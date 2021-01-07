// Copyright (c) 2016 Stefan Guthe / GCC / TU-Darmstadt. All rights reserved. 
// Use of this source code is governed by the BSD 3-Clause license that can be
// found in the LICENSE file.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

// some missing intrinsics
#ifndef __bfe
__device__ __forceinline__ unsigned int __bfe(unsigned int in, unsigned int start, unsigned int len)
{
	unsigned int out;
	asm("bfe.u32 %0, %1, %2, %3;" : "=r"(out) : "r"(in), "r"(start), "r"(len));
	return out;
}

__device__ __forceinline__ uint64 __bfe(uint64 in, unsigned int start, unsigned int len)
{
	uint64 out;
	asm("bfe.u64 %0, %1, %2, %3;" : "=l"(out) : "l"(in), "r"(start), "r"(len));
	return out;
}
#endif

#ifndef __bfi
__device__ __forceinline__ unsigned int __bfi(unsigned int a, unsigned int b, unsigned int start, unsigned int len)
{
	unsigned int out;
	asm("bfi.b32 %0, %1, %2, %3, %4;" : "=r"(out) : "r"(a), "r"(b), "r"(start), "r"(len));
	return out;
}

__device__ __forceinline__ uint64 __bfi(uint64 a, uint64 b, unsigned int start, unsigned int len)
{
	uint64 out;
	asm("bfi.b64 %0, %1, %2, %3, %4;" : "=l"(out) : "l"(a), "l"(b), "r"(start), "r"(len));
	return out;
}
#endif

#ifndef __max
#define __max(a,b)  (((a) > (b)) ? (a) : (b))
#endif

#ifndef __min
#define __min(a,b)  (((a) < (b)) ? (a) : (b))
#endif

#ifndef __laneid
__device__ __forceinline__ unsigned int __laneid() {
	int ret;
	asm("mov.u32 %0, %laneid;" : "=r"(ret) : );
	return ret;
}
#endif
