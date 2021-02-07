#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#include "helper_math.h"
#include "RendererRBUC8x8_helper.h"

/*** comment this out if you want to use CPU functions ***/
#define OWN_FUNCTIONS

// reserve constant memory for swizzle indices
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
// functions, which start kernels from host side (definitions below)
template<class T>
void SubtractMin_GPU(T *raw, T *swizzle, T min);

template<class T>
void SubtractMax_GPU(T *raw, T *swizzle, T max);

template<class T>
void TransformGradient_GPU(T *raw, T *swizzle, T min, T max);

template<class T, class T2>
void TransformHaar_GPU(T *raw, T2 *swizzle, T min, T max);

/**************************************************************/


/*********************** CUDA kernel **************************/
template<class T>
__global__ void SubtractMin_kernel(T *raw, T *swizzle, T min)
{
	int x = threadIdx.x;

	__shared__ T delta[64];
	delta[x] = raw[x] - min;

	// we need a fully written delta
	__syncthreads();

	SwizzleRegular_kernel(delta, swizzle);
}

template<class T>
__global__ void SubtractMax_kernel(T *raw, T *swizzle, T max)
{
	int x = threadIdx.x;

	__shared__ T delta[64];
	delta[x] = max - raw[x];

	// we need a fully written delta
	__syncthreads();

	SwizzleRegular_kernel(delta, swizzle);
}

template<class T>
__global__ void TransformGradient_kernel(T *raw, T *swizzle, T min, T max)
{
	// TODO
	__shared__ T delta[64]; 
	T avg = getAvg_Device(min, max);
	unsigned int x = threadIdx.x;
	unsigned int y = threadIdx.y;
	unsigned int z = threadIdx.z;
	unsigned int idx = x + 4 * (y + 4 * z);
	T pred = Predict_Device(raw, avg, min, max, x, y, z, idx);
	delta[idx] = Encode_Device(pred, min, max, raw[idx]);
	// we need a fully written delta
	__syncthreads();

	SwizzleRegular_kernel(delta, swizzle);
}

/*** Works right now only with 8 bit data (all benchmark data is 8 bit) ***/
template<class T, class T2>
__global__ void TransformHaar_kernel(T *raw, T2 *swizzle, T min, T max)
{
	// TODO
	T avg = getAvg_Device(min, max);
	__shared__ T2 delta[64];
	__shared__ int dlt[64];
	unsigned int x = threadIdx.x;
	unsigned int y = threadIdx.y;
	unsigned int z = threadIdx.z;
	unsigned int idx = x + 4 * (y + 4 * z);

	__syncthreads();

	dlt[idx] = (int)raw[idx] - avg;

	__syncthreads();

	if (x % 2 == 0) {
		dlt[idx + 1] -= dlt[idx];
		dlt[idx] += dlt[idx + 1] >> 1;
	}

	__syncthreads();

	if (y % 2 == 0) {
		dlt[idx + 4] -= dlt[idx];
		dlt[idx] += dlt[idx + 4] >> 1;
	}

	__syncthreads();

	if (z % 2 == 0) {
		dlt[idx + 16] -= dlt[idx];
		dlt[idx] += dlt[idx + 16] >> 1;
	}

	__syncthreads();

	if (x == 0 && (y % 2) == 0 && (z % 2) == 0) {
		dlt[idx + 2] -= dlt[idx];
		dlt[idx] += dlt[idx + 2] >> 1;
	}

	__syncthreads();

	if ((x % 2) == 0 && y == 0 && (z % 2) == 0) {
		dlt[idx + 8] -= dlt[idx];
		dlt[idx] += dlt[idx + 8] >> 1;
	}
	__syncthreads();

	if ((x % 2) == 0 && (y % 2) == 0 && z == 0) {
		dlt[idx + 32] -= dlt[idx];
		dlt[idx] += dlt[idx + 32] >> 1;
	}

	__syncthreads();

	if (dlt[idx] >= 0) delta[idx] = dlt[idx] << 1;
	else delta[idx] = (dlt[idx] << 1) ^ 0xffffffff;

	__syncthreads();

	// printf("delta: %d \n", swizzle[idx]);
	SwizzleWavelet_kernel(delta, swizzle);
}

/******************************************************************************/


/************************** DEVICE FUNCTIONS *********************************/
template<class T>
__device__ void SwizzleWavelet_kernel(T *delta, T *swizzled)
{
	int idx;
	if (blockDim.y == 0 && blockDim.z == 0)
		idx = threadIdx.x;
	else
		idx = threadIdx.x + 4 * (threadIdx.y + 4 * threadIdx.z);

	swizzled[swizzle_64_GPU[idx]] = delta[idx];
}

template<class T>
__device__ void SwizzleRegular_kernel(T *delta, T *swizzled)
{
	int idx;
	if (blockDim.y == 0 && blockDim.z == 0)
		idx = threadIdx.x;
	else
		idx = threadIdx.x + 4 * (threadIdx.y + 4 * threadIdx.z);
	swizzled[swizzle_32_GPU[idx & 31] + (idx & 32)] = delta[idx];
}

template <class T> 
__device__ T getAvg_Device(T &a, T & b);


/*
*
*	Here are a lot of template specific code for short, char, short4 and char4 datatypes.
*	The current three data sets (bucky, present, tree) are loaded in as 8 bits, so many
*	of those functions are never called. Because they exist in RendererRBUC8x8_64.cu, we
*	also put them here. So this is pretty much a copy past (we couldve also just declare those
*   functions as device and host functions but for sake of clarity we copied them into this file)
*/
template <> 
__device__ unsigned char getAvg_Device(unsigned char &a, unsigned char &b) { return (unsigned char)(((int)a + (int)b + 1) >> 1); }

template <> 
__device__ unsigned short getAvg_Device(unsigned short &a, unsigned short &b) { return (unsigned short)(((int)a + (int)b + 1) >> 1); }

template <> 
__device__ uchar4 getAvg_Device(uchar4 &a, uchar4 &b) { return make_uchar4(getAvg_Device(a.x, b.x), getAvg_Device(a.y, b.y), getAvg_Device(a.z, b.z), getAvg_Device(a.w, b.w)); }

template <> 
__device__ ushort4 getAvg_Device(ushort4 &a, ushort4 &b) { return make_ushort4(getAvg_Device(a.x, b.x), getAvg_Device(a.y, b.y), getAvg_Device(a.z, b.z), getAvg_Device(a.w, b.w)); }

__device__ unsigned char Predict_Device(unsigned char *raw, unsigned char avg, unsigned char min,
	unsigned char max, unsigned int x, unsigned int y, unsigned int z, unsigned int idx)
{
	int pred = 0;
	if ((x == 0) && (y == 0) && (z == 0)) pred = avg;
	if (x > 0) pred += raw[idx - 1];
	if (y > 0) pred += raw[idx - 4];
	if ((x > 0) && (y > 0)) pred -= raw[idx - 5];
	if (z > 0) pred += raw[idx - 16];
	if ((x > 0) && (z > 0)) pred -= raw[idx - 17];
	if ((y > 0) && (z > 0)) pred -= raw[idx - 20];
	if ((x > 0) && (y > 0) && (z > 0)) pred += raw[idx - 21];
	if (pred < min) return min;
	if (pred > max) return max;
	return (unsigned char)pred;
}

__device__ unsigned short Predict_Device(unsigned short *raw, unsigned short avg, unsigned short min,
	unsigned short max, unsigned int x, unsigned int y, unsigned int z, unsigned int idx)
{
	int pred = 0;
	if ((x == 0) && (y == 0) && (z == 0)) pred = avg;
	if (x > 0) pred += raw[idx - 1];
	if (y > 0) pred += raw[idx - 4];
	if ((x > 0) && (y > 0)) pred -= raw[idx - 5];
	if (z > 0) pred += raw[idx - 16];
	if ((x > 0) && (z > 0)) pred -= raw[idx - 17];
	if ((y > 0) && (z > 0)) pred -= raw[idx - 20];
	if ((x > 0) && (y > 0) && (z > 0)) pred += raw[idx - 21];
	if (pred < min) return min;
	if (pred > max) return max;
	return (unsigned short)pred;
}

__device__ uchar4 Predict_Device(uchar4 *raw, uchar4 avg, uchar4 min, uchar4 max,
	unsigned int x, unsigned int y, unsigned int z, unsigned int idx)
{
	int4 pred = make_int4(0, 0, 0, 0);
	if ((x == 0) && (y == 0) && (z == 0)) pred = make_int4(avg.x, avg.y, avg.z, avg.w);
	if (x > 0) pred += raw[idx - 1];
	if (y > 0) pred += raw[idx - 4];
	if ((x > 0) && (y > 0)) pred -= raw[idx - 5];
	if (z > 0) pred += raw[idx - 16];
	if ((x > 0) && (z > 0)) pred -= raw[idx - 17];
	if ((y > 0) && (z > 0)) pred -= raw[idx - 20];
	if ((x > 0) && (y > 0) && (z > 0)) pred += raw[idx - 21];
	if (pred.x < min.x) pred.x = min.x;
	if (pred.y < min.y) pred.y = min.y;
	if (pred.z < min.z) pred.z = min.z;
	if (pred.w < min.w) pred.w = min.w;
	if (pred.x > max.x) pred.x = max.x;
	if (pred.y > max.y) pred.y = max.y;
	if (pred.z > max.z) pred.z = max.z;
	if (pred.w > max.w) pred.w = max.w;
	return make_uchar4(pred.x, pred.y, pred.z, pred.w);
}

__device__ ushort4 Predict_Device(ushort4 *raw, ushort4 avg, ushort4 min, ushort4 max,
	unsigned int x, unsigned int y, unsigned int z, unsigned int idx)
{
	int4 pred = make_int4(0, 0, 0, 0);
	if ((x == 0) && (y == 0) && (z == 0)) pred = make_int4(avg.x, avg.y, avg.z, avg.w);
	if (x > 0) pred += raw[idx - 1];
	if (y > 0) pred += raw[idx - 4];
	if ((x > 0) && (y > 0)) pred -= raw[idx - 5];
	if (z > 0) pred += raw[idx - 16];
	if ((x > 0) && (z > 0)) pred -= raw[idx - 17];
	if ((y > 0) && (z > 0)) pred -= raw[idx - 20];
	if ((x > 0) && (y > 0) && (z > 0)) pred += raw[idx - 21];
	if (pred.x < min.x) pred.x = min.x;
	if (pred.y < min.y) pred.y = min.y;
	if (pred.z < min.z) pred.z = min.z;
	if (pred.w < min.w) pred.w = min.w;
	if (pred.x > max.x) pred.x = max.x;
	if (pred.y > max.y) pred.y = max.y;
	if (pred.z > max.z) pred.z = max.z;
	if (pred.w > max.w) pred.w = max.w;
	return make_ushort4(pred.x, pred.y, pred.z, pred.w);
}

template<class T>
__device__ T EncodeInternal_Device(T pred, T min_v, T max, T raw) 
{
	int max_pos = (int)max - pred;
	int max_neg = (int)min_v - pred;
	int dlt = (int)raw - pred;
	// -max_neg = max_pos + 1 is the balanced case
	int m = (dlt < 0) ? -1 : 0;
	int balanced_max = min(max_neg ^ -1, max_pos);
	// balanced_max can be -1 if max_neg is 0
	if ((dlt ^ m) > balanced_max)
	{
		// off balance
		return (dlt ^ m) + (balanced_max + 1);
	}
	else
	{
		// balanced
		return (dlt << 1) ^ m;
	}
}

template <class T>
__device__ T Encode_Device(T& pred, T& min, T& max, T& raw);

template<>
__device__ unsigned char Encode_Device(unsigned char& pred, unsigned char& min, unsigned char& max, unsigned char& raw)
{ 
	return EncodeInternal_Device<unsigned char>(pred, min, max, raw); 
}

template<> 
__device__ unsigned short Encode_Device(unsigned short& pred, unsigned short& min, unsigned short& max, unsigned short& raw)
{
	return EncodeInternal_Device<unsigned short>(pred, min, max, raw); 
}

template<>
__device__ uchar4 Encode_Device(uchar4& pred, uchar4& min, uchar4& max, uchar4& raw)
{
	return make_uchar4(
		EncodeInternal_Device<unsigned char>(pred.x, min.x, max.x, raw.x),
		EncodeInternal_Device<unsigned char>(pred.y, min.y, max.y, raw.y),
		EncodeInternal_Device<unsigned char>(pred.z, min.z, max.z, raw.z),
		EncodeInternal_Device<unsigned char>(pred.w, min.w, max.w, raw.w));
}

template<> 
__device__ ushort4 Encode_Device(ushort4& pred, ushort4& min, ushort4& max, ushort4& raw)
{
	return make_ushort4(
		EncodeInternal_Device<unsigned short>(pred.x, min.x, max.x, raw.x),
		EncodeInternal_Device<unsigned short>(pred.y, min.y, max.y, raw.y),
		EncodeInternal_Device<unsigned short>(pred.z, min.z, max.z, raw.z),
		EncodeInternal_Device<unsigned short>(pred.w, min.w, max.w, raw.w));
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
	T *raw_GPU;
	T *swizzle_GPU;
	cudaMalloc((void**)&raw_GPU, sizeof(T) * 64);
	cudaMalloc((void**)&swizzle_GPU, sizeof(T) * 64);
	cudaMemcpy(raw_GPU, raw, sizeof(T) * 64, cudaMemcpyHostToDevice);
	dim3 threadBlock(4, 4, 4);
	TransformGradient_kernel<T> << <1, threadBlock >> >(raw_GPU, swizzle_GPU, min, max);

	cudaMemcpy(swizzle, swizzle_GPU, sizeof(T) * 64, cudaMemcpyDeviceToHost);

	cudaFree(raw_GPU);
	cudaFree(swizzle_GPU);
}

template<class T, class T2>
void TransformHaar_GPU(T *raw, T2 *swizzle, T min, T max)
{
	T *raw_GPU;
	T2 *swizzle_GPU;
	// printf("T2: %d \n", sizeof(T2));
	cudaMalloc((void**)&raw_GPU, sizeof(T) * 64);
	cudaMalloc((void**)&swizzle_GPU, sizeof(T2) * 64);
	cudaMemcpy(raw_GPU, raw, sizeof(T) * 64, cudaMemcpyHostToDevice);

	dim3 threadBlock(4, 4, 4);
	TransformHaar_kernel<T> << <1, threadBlock >> >(raw_GPU, swizzle_GPU, min, max);

	cudaMemcpy(swizzle, swizzle_GPU, sizeof(T2) * 64, cudaMemcpyDeviceToHost);

	cudaFree(raw_GPU);
	cudaFree(swizzle_GPU);

}

/*******************************************************************************************/



// WIP
/*
*	Rewrite the above code to do following:
*	Create a grid (or even better: use streams to have better overlap, but then you need pinned memory)
*	which distributes all eligible bricks onto the GPU and computes them in parallel. 
*	Computation contains:
*	    - load the brick per thread block into shared memory
*		- apply all prediction functions + swizzle on them and decides which is the most efficient
*		- encode the now predicted brick with RBUC
*       - transfer the bricks back to the host side
*       - (also maybe update the global statistics but those counter are right now only available on host side)
*
*	After implementing, use nvprof to evaluate
*/

/*

template<class T>
int* Prediction_GPU(T *raw, T min, T max)
{
	int size[4];

	T *raw_GPU;
	int *size_GPU;


	cudaMalloc((void**)&raw_GPU, sizeof(T) * 64);
	cudaMalloc((void**)&size_GPU, sizeof(int) * 4);
	cudaMemcpy(raw_GPU, raw, sizeof(T) * 64, cudaMemcpyHostToDevice);

	dim threadBlock(4, 4, 4);
	Prediction_Kernel<T> << <1, threadBlock >> >(raw_GPU, min, max, size);

	cudaMemcpy(size, size_GPU, sizeof(int) * 4, cudaMemcpyHostToDevice);

	cudaFree(raw_GPU);
	cudaFree(size_GPU);

	return size;
}

template<class T>
__global__ void Prediction_Kernel(T *raw, T min, T max, int* result)
{
	unsigned int x = threadIdx.x;
	unsigned int y = threadIdx.y;
	unsigned int z = threadIdx.z;
	unsigned int idx = x + 4 * (y + 4 * z);

	// reserve shared memory
	__shared__ T smem_raw[64];
	smem_raw[idx] = raw[idx];

	__syncthreads();

	// call all predicator functions
	result[0] = SubtractMin_device(smem_raw, min);
	result[1] = SubtractMax_device(smem_raw, max);
	result[2] = TransformGradient_device(smem_raw, min, max);
	result[3] = TransformHaar_device(smem_raw, min, max);
}

template<class T>
__device__ int SubtractMin_device(T *raw, T min)
{
	int x = threadIdx.x;
	T delta[64];
	delta[x] = raw[x] - min;

	// we need a fully written delta
	__syncthreads();

	SwizzleRegular_kernel(delta, swizzle);
}

template<class T>
__device__ int SubtractMax_device(T *raw, T max)
{
	int x = threadIdx.x;
	T delta[64];
	delta[x] = max - raw[x];

	// we need a fully written delta
	__syncthreads();

	SwizzleRegular_kernel(delta, swizzle);
}

template<class T>
__device__ int TransformGradient_device(T *raw, T min, T max)
{
	// TODO
	T delta[64];
	T avg = getAvg_Device(min, max);
	unsigned int x = threadIdx.x;
	unsigned int y = threadIdx.y;
	unsigned int z = threadIdx.z;
	unsigned int idx = x + 4 * (y + 4 * z);
	T pred = Predict_Device(raw, avg, min, max, x, y, z, idx);
	delta[idx] = Encode_Device(pred, min, max, raw[idx]);
	// we need a fully written delta
	__syncthreads();

	SwizzleRegular_kernel(delta, swizzle);
}

template<class T>
__device__ void TransformHaar_device(T *raw, T min, T max)
{
	// TODO
	T avg = getAvg_Device(min, max);
	T2 delta[64];
	int dlt[64];
	unsigned int x = threadIdx.x;
	unsigned int y = threadIdx.y;
	unsigned int z = threadIdx.z;
	unsigned int idx = x + 4 * (y + 4 * z);

	dlt[idx] = (int)raw[idx] - avg;

	__syncthreads();

	if (x % 2 == 0) {
		dlt[idx + 1] -= dlt[idx];
		dlt[idx] += dlt[idx + 1] >> 1;
	}

	__syncthreads();

	if (y % 2 == 0) {
		dlt[idx + 4] -= dlt[idx];
		dlt[idx] += dlt[idx + 4] >> 1;
	}

	__syncthreads();

	if (z % 2 == 0) {
		dlt[idx + 16] -= dlt[idx];
		dlt[idx] += dlt[idx + 16] >> 1;
	}

	__syncthreads();

	if (x == 0 && (y % 2) == 0 && (z % 2) == 0) {
		dlt[idx + 2] -= dlt[idx];
		dlt[idx] += dlt[idx + 2] >> 1;
	}

	__syncthreads();

	if ((x % 2) == 0 && y == 0 && (z % 2) == 0) {
		dlt[idx + 8] -= dlt[idx];
		dlt[idx] += dlt[idx + 8] >> 1;
	}
	__syncthreads();

	if ((x % 2) == 0 && (y % 2) == 0 && z == 0) {
		dlt[idx + 32] -= dlt[idx];
		dlt[idx] += dlt[idx + 32] >> 1;
	}

	__syncthreads();

	if (dlt[idx] >= 0) delta[idx] = dlt[idx] << 1;
	else delta[idx] = (dlt[idx] << 1) ^ 0xffffffff;

	__syncthreads();
	SwizzleWavelet_kernel(delta, swizzle);
}

*/