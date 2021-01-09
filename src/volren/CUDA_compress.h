#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#include "helper_math.h"

/*** comment this out if you want to use CPU functions ***/
#define OWN_FUNCTIONS

__constant__ uint swizzle_64_GPU[64] = {
	0x00, 0x08, 0x01, 0x09,
	0x10, 0x18, 0x11, 0x19,
	0x02, 0x0a, 0x03, 0x0b,
	0x12, 0x1a, 0x13, 0x1b,

	0x20, 0x28, 0x21, 0x29,
	0x30, 0x38, 0x31, 0x39,
	0x22, 0x2a, 0x23, 0x2b,
	0x32, 0x3a, 0x33, 0x3b,

	0x04, 0x0c, 0x05, 0x0d,
	0x14, 0x1c, 0x15, 0x1d,
	0x06, 0x0e, 0x07, 0x0f,
	0x16, 0x1e, 0x17, 0x1f,

	0x24, 0x2c, 0x25, 0x2d,
	0x34, 0x3c, 0x35, 0x3d,
	0x26, 0x2e, 0x27, 0x2f,
	0x36, 0x3e, 0x37, 0x3f };

__constant__ uint swizzle_32_GPU[32] = {
	0x00, 0x01, 0x08, 0x09,
	0x02, 0x03, 0x0a, 0x0b,
	0x10, 0x11, 0x18, 0x19,
	0x12, 0x13, 0x1a, 0x1b,

	0x04, 0x05, 0x0c, 0x0d,
	0x06, 0x07, 0x0e, 0x0f,
	0x14, 0x15, 0x1c, 0x1d,
	0x16, 0x17, 0x1e, 0x1f };



/******************** Function declarations *********************/
template<class T>
void SubtractMin_GPU(T *raw, T *swizzle, T min);

template<class T>
void SubtractMax_GPU(T *raw, T *swizzle, T max);

template<class T>
void TransformGradient_GPU(T *raw, T *swizzle, T min, T max);

template<class T, class T2>
void TransformHaar_GPU(T *raw, T *swizzle, T min, T max);

/**************************************************************/


/*********************** CUDA kernel **************************/
template<class T>
__global__ void SubtractMin_kernel(T *raw, T *swizzle, T min)
{
	int x = threadIdx.x;
	T delta[64];
	delta[x] = raw[x] - min;

	// we need a fully written delta
	__syncthreads();

	SwizzleRegular_kernel(delta, swizzle);
}

template<class T>
__global__ void SubtractMax_kernel(T *raw, T *swizzle, T max)
{
	int x = threadIdx.x;
	T delta[64];
	delta[x] = max - raw[x];

	// we need a fully written delta
	__syncthreads();

	SwizzleRegular_kernel(delta, swizzle);
}

template<class T>
__global__ void TransformGradient_kernel(T *raw, T *delta, T *swizzle, T min, T max)
{
	// TODO
	T avg = ((int)min + (int)max + 1) >> 1;

	// those values are concluded from code of RendererRBUC8x8.cu, function transformGradient
	int x = threadIdx.x % 4;
	int y = (threadIdx.x / 4) % 4;
	int z = (threadIdx.x / 16) % 4;

	// we need a fully written delta
	__syncthreads();

	SwizzleRegular_kernel(delta, swizzle);
}

/*** KERNEL PRODUCES WRONG RESULTS!!! ***/
template<class T, class T2>
__global__ void TransformHaar_kernel(T *raw, T2 *delta, T2 *swizzle, T min, T max)
{
	// TODO
	T avg = ((int)min + (int)max + 1) >> 1;
	int dlt[64];
	int idx = threadIdx.x;

	dlt[idx] = (int)raw[idx] - avg;
	__syncthreads();

	if (idx % 2 == 0)
	{
		dlt[idx + 1] -= dlt[idx];
		dlt[idx] += dlt[idx + 1] >> 1;
	}
	__syncthreads();

	if ((idx / 4) % 2 == 0)
	{
		dlt[idx + 4] -= dlt[idx];
		dlt[idx] += dlt[idx + 4] >> 1;
	}
	__syncthreads();

	if ((idx / 16) % 2 == 0)
	{
		dlt[idx + 16] -= dlt[idx];
		dlt[idx] += dlt[idx + 16] >> 1;
	}
	__syncthreads();

	// second transform
	if ( ((idx % 4) == 0) && (((idx / 4) % 2) == 0) && (((idx / 16) % 2) == 0))
	{
		dlt[idx + 2] -= dlt[idx];
		dlt[idx] += dlt[idx + 2] >> 1;
	}
	__syncthreads();

	if (((idx % 2) == 0) && (((idx / 4) % 4) == 0) && (((idx / 16) % 2) == 0))
	{
		dlt[idx + 8] -= dlt[idx];
		dlt[idx] += dlt[idx + 8] >> 1;
	}
	__syncthreads();

	if (((idx % 2) == 0) && (((idx / 4) % 2) == 0) && (((idx / 16) % 4) == 0))
	{
		dlt[idx + 32] -= dlt[idx];
		dlt[idx] += dlt[idx + 32] >> 1;
	}
	__syncthreads();

	if (dlt[idx] >= 0) 
		delta[idx] = dlt[idx] << 1;
	else 
		delta[idx] = (dlt[idx] << 1) ^ 0xffffffff;

	__syncthreads();
	SwizzleWavelet_kernel(delta, swizzle);
}

/******************************************************************************/


/************************** DEVICE FUNCTIONS *********************************/
template<class T>
__device__ void SwizzleWavelet_kernel(T *delta, T *swizzled)
{
	int x = threadIdx.x;
	swizzled[swizzle_64_GPU[x]] = delta[x];
}

template<class T>
__device__ void SwizzleRegular_kernel(T *delta, T *swizzled)
{
	int x = threadIdx.x;
	swizzled[swizzle_32_GPU[x & 31] + (x & 32)] = delta[x];
}

/******************************************************************************/


/************************** Function definitions ******************************/
template<class T>
void SubtractMin_GPU(T *raw, T* swizzle, T min)
{
	T *raw_GPU;
	T *swizzle_GPU;
	cudaMalloc((void**)&raw_GPU, sizeof(T) * 64);
	cudaMalloc((void**)&swizzle_GPU, sizeof(T) * 64);
	cudaMemcpy(raw_GPU, raw, sizeof(T) * 64, cudaMemcpyHostToDevice);

	SubtractMin_kernel<T> << <1, 64 >> >(raw_GPU, swizzle_GPU, min);

	cudaMemcpy(swizzle, swizzle_GPU, sizeof(T) * 64, cudaMemcpyDeviceToHost);

	cudaFree(raw_GPU);
	cudaFree(swizzle_GPU);
}

template<class T>
void SubtractMax_GPU(T *raw, T* swizzle, T max)
{
	T *raw_GPU;
	T *swizzle_GPU;
	cudaMalloc((void**)&raw_GPU, sizeof(T) * 64);
	cudaMalloc((void**)&swizzle_GPU, sizeof(T) * 64);
	cudaMemcpy(raw_GPU, raw, sizeof(T) * 64, cudaMemcpyHostToDevice);

	SubtractMax_kernel<T> << <1, 64 >> >(raw_GPU, swizzle_GPU, max);

	cudaMemcpy(swizzle, swizzle_GPU, sizeof(T) * 64, cudaMemcpyDeviceToHost);

	cudaFree(raw_GPU);
	cudaFree(swizzle_GPU);
}

template<class T>
void TransformGradient_GPU(T *raw, T *swizzle, T min, T max)
{

}

/*** FUNCTION PRODUCES WRONG VALUES!!! ***/
template<class T, class T2>
void TransformHaar_GPU(T *raw, T2 *swizzle, T min, T max)
{
	T *raw_GPU;
	T2 *delta_GPU;
	T2 *swizzle_GPU;
	cudaMalloc((void**)&raw_GPU, sizeof(T) * 64);
	cudaMalloc((void**)&delta_GPU, sizeof(T) * 64);
	cudaMalloc((void**)&swizzle_GPU, sizeof(T) * 64);
	cudaMemcpy(raw_GPU, raw, sizeof(T) * 64, cudaMemcpyHostToDevice);

	TransformHaar_kernel<T> << <1, 64 >> >(raw_GPU, delta_GPU, swizzle_GPU, min, max);

	cudaMemcpy(swizzle, swizzle_GPU, sizeof(T) * 64, cudaMemcpyDeviceToHost);

	cudaFree(raw_GPU);
	cudaFree(delta_GPU);
	cudaFree(swizzle_GPU);

}

/*******************************************************************************************/