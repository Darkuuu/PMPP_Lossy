// Copyright (c) 2016 Stefan Guthe / GCC / TU-Darmstadt. All rights reserved. 
// Use of this source code is governed by the BSD 3-Clause license that can be
// found in the LICENSE file.

#define _CRT_SECURE_NO_WARNINGS

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include "global_defines.h"
#include "../../config.h"

// CUDA Runtime, Interop, and includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>

// CUDA utilities
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

// Helper functions
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_timer.h>

#include "BasicRenderer.h"
#include "RendererRBUC8x8.h"
#include "volumeReader.h"
#include "CompressRBUC.h"

#include <chrono>

#include "png_image.h"
#include "jpeg_image.h"
#include "pnm_image.h"

#include <set>

#ifdef PVM_SUPPORT
#include "../v3/ddsbase.h"
#endif


#define MAX_EPSILON_ERROR 5.00f
#define THRESHOLD         0.30f

// compression measurement hook
bool displayEntropy;

// this has quite some performance impact, even if disabled
//#define LEARNING

// Define the files that are to be save and the reference images for validation
const char *sOriginal[] =
{
	"volume.ppm",
	NULL
};

const char *sReference[] =
{
	"ref_volume.ppm",
	NULL
};

const char *sSDKsample = "CUDA 3D Volume Render";

cudaExtent volumeSize = make_cudaExtent(32, 32, 32);
cudaExtent originalSize = make_cudaExtent(0, 0, 0);
float3 g_scale;
unsigned int g_components;
bool g_do_snapshot = false;

const char *volumeFilename = "stagbeetle-208x208x123.dat.gz";

Renderer<VolumeType8> *renderer8;
Renderer<VolumeType16> *renderer16;
Renderer<VolumeType32> *renderer32;
Renderer<VolumeType64> *renderer64;
int volumeType = 0;
bool g_automode = false;
FILE *g_logfile;

float4 *transferFunc = 0;
size_t transferFunc_size = 0;
int transferFuncMapping[255];
bool transferFuncMappingActive = false;
int transfer = 0;
bool smooth_transfer = false;

#define COMPRESS_RBUC_GRADIENT

uint width = 512, height = 512;
dim3 blockSize(16, 16);
dim3 gridSize;

float3 viewRotation;
float3 viewTranslation = make_float3(0.0, 0.0, -4.0f);
float viewMatrix[12];
float invViewMatrix[12];

float density = 0.05f;
float brightness = 1.0f;
float transferOffset = 0.0f;
float transferScale = 1.0f;
bool linearFiltering = true;
bool backgroundWhite = false;
bool rgbColor = false;
bool lighting = false;
unsigned int auto_mode = 0;
bool basic = false;
bool fixed = false;
bool learning = false;

GLuint pbo = 0;     // OpenGL pixel buffer object
GLuint tex = 0;     // OpenGL texture object
struct cudaGraphicsResource *cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)

StopWatchInterface *timer = 0;

// Auto-Verification Code
const int frameCheckNumber = 2;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
bool g_fast = false;
unsigned int frameCount = 0;
float tstep = 0.01f;

int *pArgc;
char **pArgv;

#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif

void initPixelBuffer();

void computeFPS()
{
	frameCount++;
	fpsCount++;

	if (fpsCount == fpsLimit)
	{
		char fps[256];
		//float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
		float ifps = 1.f / (sdkGetTimerValue(&timer) / (fpsLimit * 1000.f));
		sprintf(fps, "Volume Render: %3.1f fps", ifps);

		glutSetWindowTitle(fps);
		fpsCount = 0;

		//fpsLimit = (int)MAX(1.f, MIN(10.f, ifps));
		fpsLimit = (int)MAX(1.f, ifps);
		sdkResetTimer(&timer);
	}
}

int iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

static double factor_x = 0.0;
static double factor_y = 0.0;
static double factor_z = 0.0;
static double factor_t = 0.0;
static double factor_w = 0.0;
static int map_count[21 * 21] = {
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

void resetTF()
{
	if (transferFunc != NULL)
	{
		delete[] transferFunc;
		transferFunc = NULL;
	}
	int flt_size = 0;
	switch (transfer & 255)
	{
	case 1:
		transferFunc_size = 4096;
		transferFunc = new float4[transferFunc_size];
		for (int i = 0; i < 19; i++) transferFunc[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		for (int i = 19; i < 52; i++) transferFunc[i] = make_float4(1.0f, 1.0f, 1.0f, 0.2f);
		for (int i = 52; i < 175; i++) transferFunc[i] = make_float4(0.0f, 1.0f, 0.0f, 0.3f);
		for (int i = 175; i < 666; i++) transferFunc[i] = make_float4(0.8f, 0.4f, 0.0f, 0.4f);
		for (int i = 666; i < 1034; i++) transferFunc[i] = make_float4(1.0f, 1.0f, 1.0f, 0.5f);
		for (int i = 1034; i < 1157; i++) transferFunc[i] = make_float4(0.8f, 0.5f, 0.2f, 0.6f);
		for (int i = 1157; i < 1485; i++) transferFunc[i] = make_float4(0.5f, 0.0f, 0.0f, 0.7f);
		for (int i = 1485; i < 1853; i++) transferFunc[i] = make_float4(0.5f, 0.5f, 1.0f, 0.8f);
		for (int i = 1853; i < 2222; i++) transferFunc[i] = make_float4(0.8f, 0.6f, 0.4f, 0.9f);
		for (int i = 2222; i < 4096; i++) transferFunc[i] = make_float4(0.8f, 0.8f, 1.0f, 1.0f);
		flt_size = 31;
		break;
	case 2:
		transferFunc_size = 4096;
		transferFunc = new float4[transferFunc_size];
		for (int i = 0; i < 41; i++) transferFunc[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		for (int i = 41; i < 74; i++) transferFunc[i] = make_float4(1.0f, 1.0f, 1.0f, 0.2f);
		for (int i = 74; i < 197; i++) transferFunc[i] = make_float4(0.0f, 1.0f, 0.0f, 0.3f);
		for (int i = 197; i < 688; i++) transferFunc[i] = make_float4(0.8f, 0.4f, 0.0f, 0.4f);
		for (int i = 688; i < 1056; i++) transferFunc[i] = make_float4(1.0f, 1.0f, 1.0f, 0.5f);
		for (int i = 1046; i < 1179; i++) transferFunc[i] = make_float4(0.8f, 0.5f, 0.2f, 0.6f);
		for (int i = 1179; i < 1507; i++) transferFunc[i] = make_float4(0.5f, 0.0f, 0.0f, 0.7f);
		for (int i = 1507; i < 1875; i++) transferFunc[i] = make_float4(0.5f, 0.5f, 1.0f, 0.8f);
		for (int i = 1875; i < 2244; i++) transferFunc[i] = make_float4(0.8f, 0.6f, 0.4f, 0.9f);
		for (int i = 2244; i < 4096; i++) transferFunc[i] = make_float4(0.8f, 0.8f, 1.0f, 1.0f);
		flt_size = 31;
		break;
	case 3:
		transferFunc_size = 4096;
		transferFunc = new float4[transferFunc_size];
		for (int i = 0; i < 93; i++) transferFunc[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		for (int i = 93; i < 215; i++) transferFunc[i] = make_float4(1.0f, 0.0f, 0.0f, 0.1f);
		for (int i = 215; i < 871; i++) transferFunc[i] = make_float4(0.0f, 0.0f, 1.0f, 0.2f);
		for (int i = 871; i < 1526; i++) transferFunc[i] = make_float4(0.0f, 1.0f, 0.0f, 0.4f);
		for (int i = 1526; i < 2222; i++) transferFunc[i] = make_float4(1.0f, 1.0f, 1.0f, 0.6f);
		for (int i = 2222; i < 3328; i++) transferFunc[i] = make_float4(1.0f, 0.8f, 0.6f, 0.8f);
		for (int i = 3328; i < 4096; i++) transferFunc[i] = make_float4(0.8f, 0.8f, 1.0f, 1.0f);
		flt_size = 31;
		break;
	case 4:
		// carp
		transferFunc_size = 4096;
		transferFunc = new float4[transferFunc_size];
		for (int i = 0; i < 515; i++) transferFunc[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		for (int i = 515; i < 715; i++) transferFunc[i] = make_float4(0.0f, 1.0f, 0.0f, 1.0f);
		for (int i = 715; i < 915; i++) transferFunc[i] = make_float4(0.0f, 1.0f, 1.0f, 1.0f);
		for (int i = 915; i < 1115; i++) transferFunc[i] = make_float4(1.0f, 0.0f, 0.0f, 0.03f);
		for (int i = 1115; i < 1315; i++) transferFunc[i] = make_float4(1.0f, 1.0f, 0.0f, 0.1f);
		for (int i = 1315; i < 1515; i++) transferFunc[i] = make_float4(0.0f, 0.0f, 1.0f, 0.3f);
		for (int i = 1515; i < 4096; i++) transferFunc[i] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
		flt_size = 63;
		break;
	case 5:
		transferFunc_size = 4096;
		transferFunc = new float4[transferFunc_size];
		for (int i = 0; i < 93; i++) transferFunc[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		for (int i = 93; i < 215; i++) transferFunc[i] = make_float4(1.0f, 0.0f, 0.0f, 1.0f);
		for (int i = 215; i < 871; i++) transferFunc[i] = make_float4(0.0f, 0.0f, 1.0f, 0.5f * 0.5f);
		for (int i = 871; i < 1526; i++) transferFunc[i] = make_float4(0.0f, 1.0f, 0.0f, 0.25f * 0.25f);
		for (int i = 1526; i < 2222; i++) transferFunc[i] = make_float4(1.0f, 1.0f, 1.0f, 0.125f * 0.125f);
		for (int i = 2222; i < 3328; i++) transferFunc[i] = make_float4(1.0f, 0.8f, 0.6f, 0.0625f * 0.0625f);
		for (int i = 3328; i < 4096; i++) transferFunc[i] = make_float4(0.8f, 0.8f, 1.0f, 0.03125f * 0.03125f);
		flt_size = 31;
		break;
	case 255:
		transferFunc_size = 4096;
		transferFunc = new float4[transferFunc_size];
		for (int i = 0; i < 256; i++) transferFunc[i] = make_float4(0.0, 0.0, 0.0, 0.0);
		for (int i = 256; i < 4096; i++) transferFunc[i] = make_float4(1.0, 1.0, 1.0, 1.0);
		break;
	default:
		transferFunc_size = 9;
		transferFunc = new float4[transferFunc_size];
		transferFunc[0] = make_float4(0.0, 0.0, 0.0, 0.0);
		transferFunc[1] = make_float4(1.0, 0.0, 0.0, 1.0);
		transferFunc[2] = make_float4(1.0, 0.5, 0.0, 1.0);
		transferFunc[3] = make_float4(1.0, 1.0, 0.0, 1.0);
		transferFunc[4] = make_float4(0.0, 1.0, 0.0, 1.0);
		transferFunc[5] = make_float4(0.0, 1.0, 1.0, 1.0);
		transferFunc[6] = make_float4(0.0, 0.0, 1.0, 1.0);
		transferFunc[7] = make_float4(1.0, 0.0, 1.0, 1.0);
		transferFunc[8] = make_float4(0.0, 0.0, 0.0, 0.0);
	}

	if (transfer > 255)
	{
		for (int i = 0; i < transferFunc_size; i++) if (transferFunc[i].w > 0.0f) transferFunc[i].w = 1.0f;
	}

	if (smooth_transfer && (transferFunc_size > 64))
	{
		float4 *tmpFunc = new float4[transferFunc_size];

		for (int i = 0; i < transferFunc_size; i++)
		{
			float count = 0.0f;
			tmpFunc[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
			if (transferFunc[i].w > 0.0f)
			{
				for (int j = i - flt_size; j <= i + flt_size; j++)
				{
					if ((j >= 0) && (j < transferFunc_size))
					{
						if (transferFunc[j].w > 0.0f)
						{
							tmpFunc[i] += transferFunc[j];
							count += 1.0f;
						}
					}
				}
				tmpFunc[i] /= count;
			}
			else
				tmpFunc[i] = transferFunc[i];
		}

		std::swap(tmpFunc, transferFunc);
		delete[] tmpFunc;
	}

	if (transferFuncMappingActive && (transferFunc_size > 256))
	{
		float4 *tmp = new float4[256];
		for (int i = 0; i < 256; i++)
		{
			std::cout << i << " -> " << transferFuncMapping[i] << std::endl;
			tmp[i] = transferFunc[transferFuncMapping[i]];
		}

		std::swap(tmp, transferFunc);
		transferFunc_size = 256;
		delete[] tmp;
	}
}

// render image using CUDA
void render()
{
	if (volumeType == 0)
	{
		renderer8->copyViewMatrix(viewMatrix, sizeof(float4) * 3);
		renderer8->copyInvViewMatrix(invViewMatrix, sizeof(float4) * 3);
	}
	else if (volumeType == 1)
	{
		renderer16->copyViewMatrix(viewMatrix, sizeof(float4) * 3);
		renderer16->copyInvViewMatrix(invViewMatrix, sizeof(float4) * 3);
	}
	else if (volumeType == 2)
	{
		renderer32->copyViewMatrix(viewMatrix, sizeof(float4) * 3);
		renderer32->copyInvViewMatrix(invViewMatrix, sizeof(float4) * 3);
	}
	else
	{
		renderer64->copyViewMatrix(viewMatrix, sizeof(float4) * 3);
		renderer64->copyInvViewMatrix(invViewMatrix, sizeof(float4) * 3);
	}

	// map PBO to get CUDA device pointer
	uint *d_output;
	// map PBO to get CUDA device pointer
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,
		cuda_pbo_resource));
	//printf("CUDA mapped PBO: May access %ld bytes\n", num_bytes);

	// clear image
	checkCudaErrors(cudaMemset(d_output, 0, width*height * 4));

	float smallestExtent;

	float3 scale = g_scale;
	float factor;
	if (((rgbColor) || (lighting)) && (volumeType < 2) && (g_components > 1))
		scale.z *= 0.25f;

	factor = 2.0f / sqrtf(scale.x * scale.x + scale.y * scale.y + scale.z * scale.z);
	scale *= factor;

	smallestExtent = factor * std::min(std::min(g_scale.x / volumeSize.width, g_scale.y / volumeSize.height), g_scale.z / volumeSize.depth);

	if (fixed)
		tstep = 0.01f;
	else
		tstep = std::min(0.01f, smallestExtent);

	dim3 warpDim;

	// call CUDA kernel, writing results to PBO
	if (volumeType == 0)
	{
		renderer8->setScale(scale);
		renderer8->getLaunchParameter(warpDim, width, height, tstep);
	}
	else if (volumeType == 1)
	{
		renderer16->setScale(scale);
		renderer16->getLaunchParameter(warpDim, width, height, tstep);
	}
	else if (volumeType == 2)
	{
		renderer32->setScale(scale);
		renderer32->getLaunchParameter(warpDim, width, height, tstep);
	}
	else
	{
		renderer64->setScale(scale);
		renderer64->getLaunchParameter(warpDim, width, height, tstep);
	}
	if (learning)
	{
		dim3 selectedWarpDim = warpDim;
		dim3 bestWarpDim = 0;
		double bestTime = 1e26;
		double selectedTime;
		int sel_idx;
		int best_idx;
		for (unsigned int c = 0; c < (learning ? 21u : 1u); c++)
		{
#ifdef WIN32
			LARGE_INTEGER start, end;
#else
			std::chrono::high_resolution_clock::time_point start, end;
#endif
			warpDim.x = warpDim.y = 1;
			for (unsigned int i = 0; i < c; i++)
			{
				warpDim.x <<= 1;
				if (warpDim.x * warpDim.y > 32)
				{
					warpDim.x = 1;
					warpDim.y <<= 1;
				}
			}
			warpDim.z = 32 / (warpDim.x * warpDim.y);
#ifdef WIN32
			QueryPerformanceCounter(&start);
#else
			start = std::chrono::high_resolution_clock::now();
#endif

			blockSize = dim3(32, 4, 2);
			gridSize = dim3(iDivUp(width, warpDim.x * blockSize.y), iDivUp(height, warpDim.y * blockSize.z));

			if (volumeType == 0)
				renderer8->render_kernel(gridSize, blockSize, warpDim, d_output, width, height, density, brightness, transferOffset, transferScale, tstep, backgroundWhite);
			else if (volumeType == 1)
				renderer16->render_kernel(gridSize, blockSize, warpDim, d_output, width, height, density, brightness, transferOffset, transferScale, tstep, backgroundWhite);
			else if (volumeType == 2)
				renderer32->render_kernel(gridSize, blockSize, warpDim, d_output, width, height, density, brightness, transferOffset, transferScale, tstep, backgroundWhite);
			else
				renderer64->render_kernel(gridSize, blockSize, warpDim, d_output, width, height, density, brightness, transferOffset, transferScale, tstep, backgroundWhite);

			//std::cout << "warpDim = " << warpDim.x << ", " << warpDim.y << ", " << warpDim.z << std::endl;

			getLastCudaError("kernel failed");

#ifdef WIN32
			QueryPerformanceCounter(&end);
			LARGE_INTEGER f;
			QueryPerformanceFrequency(&f);
			double sec = (double)(end.QuadPart - start.QuadPart) / (double)(f.QuadPart);
			double time = sec * 1000000.0;
#else
			end = std::chrono::high_resolution_clock::now();
			double time = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
#endif
			if (time < bestTime)
			{
				bestTime = time;
				bestWarpDim = warpDim;
				best_idx = c;
			}
			if ((warpDim.x == selectedWarpDim.x) && (warpDim.y == selectedWarpDim.y) && (warpDim.z == selectedWarpDim.z))
			{
				selectedTime = time;
				sel_idx = c;
			}
			//std::cout << "warpDim = " << warpDim.x << ", " << warpDim.y << ", " << warpDim.z << ": " << time << "ns" << std::endl;
		}
		//std::cout << "warpDim = " << selectedWarpDim.x << ", " << selectedWarpDim.y << ", " << selectedWarpDim.z << ": " << selectedTime << "ns";
		//std::cout << " -> " << bestWarpDim.x << ", " << bestWarpDim.y << ", " << bestWarpDim.z << ": " << bestTime << "ns" << std::endl;
		factor_x += (double)bestWarpDim.x / (double)selectedWarpDim.x;
		factor_y += (double)bestWarpDim.y / (double)selectedWarpDim.y;
		factor_z += (double)bestWarpDim.z / (double)selectedWarpDim.z;
		factor_t += (double)bestTime / (double)selectedTime;
		factor_w += 1.0;
		std::cout << "factor: " << factor_x / factor_w << ", " << factor_y / factor_w << ", " << factor_z / factor_w << ", " << factor_t / factor_w << std::endl;
		map_count[best_idx + sel_idx * 21]++;
		for (unsigned int idx = 0; idx < 21 * 21; idx++)
		{
			if (map_count[idx] > 0)
			{
				unsigned int b = idx % 21;
				unsigned int s = idx / 21;
				warpDim.x = warpDim.y = 1;
				for (unsigned int i = 0; i < s; i++)
				{
					warpDim.x <<= 1;
					if (warpDim.x * warpDim.y > 32)
					{
						warpDim.x = 1;
						warpDim.y <<= 1;
					}
				}
				warpDim.z = 32 / (warpDim.x * warpDim.y);
				std::cout << "  [" << warpDim.x << ", " << warpDim.y << ", " << warpDim.z << "]";
				warpDim.x = warpDim.y = 1;
				for (unsigned int i = 0; i < b; i++)
				{
					warpDim.x <<= 1;
					if (warpDim.x * warpDim.y > 32)
					{
						warpDim.x = 1;
						warpDim.y <<= 1;
					}
				}
				warpDim.z = 32 / (warpDim.x * warpDim.y);
				std::cout << " -> [" << warpDim.x << ", " << warpDim.y << ", " << warpDim.z << "]: " << map_count[idx] << std::endl;
			}
		}
	}
	else
	{
		if (basic) {
			warpDim.x = 16;
			warpDim.y = 2;
			warpDim.z = 1;
			blockSize = dim3(32, 1, 8);
		}
		else
		{
			blockSize = dim3(32, 4, 2);
		}
		gridSize = dim3(iDivUp(width, warpDim.x * blockSize.y), iDivUp(height, warpDim.y * blockSize.z));

		if (volumeType == 0)
			renderer8->render_kernel(gridSize, blockSize, warpDim, d_output, width, height, density, brightness, transferOffset, transferScale, tstep, backgroundWhite);
		else if (volumeType == 1)
			renderer16->render_kernel(gridSize, blockSize, warpDim, d_output, width, height, density, brightness, transferOffset, transferScale, tstep, backgroundWhite);
		else if (volumeType == 2)
			renderer32->render_kernel(gridSize, blockSize, warpDim, d_output, width, height, density, brightness, transferOffset, transferScale, tstep, backgroundWhite);
		else
			renderer64->render_kernel(gridSize, blockSize, warpDim, d_output, width, height, density, brightness, transferOffset, transferScale, tstep, backgroundWhite);

		//std::cout << "warpDim = " << warpDim.x << ", " << warpDim.y << ", " << warpDim.z << std::endl;
		//std::cout << "gridSize = " << gridSize.x << ", " << gridSize.y << ", " << gridSize.z << std::endl;
		//std::cout << "blockSize = " << blockSize.x << ", " << blockSize.y << ", " << blockSize.z << std::endl;
		//std::cout << "threads = " << gridSize.x * gridSize.y * gridSize.z * blockSize.x * blockSize.y * blockSize.z << std::endl;
		//std::cout << "pixel   = " << (gridSize.x * gridSize.y * gridSize.z * blockSize.x * blockSize.y * blockSize.z) / warpDim.z << std::endl;

		getLastCudaError("kernel failed");
	}

	if (g_do_snapshot)
	{
		unsigned char *buffer = new unsigned char[width*height*4];
		checkCudaErrors(cudaMemcpy(buffer, d_output, width*height * 4, cudaMemcpyDeviceToHost));

#ifdef LIBPNG_SUPPORT
		PngImage out;
		out.SetBitDepth(8);
#else
#ifdef LIBJPEG_SUPPORT
		JpegImage out;
		out.SetQuality(100);
#else
		PnmImage out;
		out.SetBitDepth(8);
#endif
#endif
		out.SetWidth(width);
		out.SetHeight(height);
		out.SetComponents(3);

		for (unsigned int y = 0; y < height; y++)
		{
			for (unsigned int x = 0; x < width; x++)
			{
				for (unsigned int c = 0; c < 3; c++)
				{
					out.VSetValue(x, height - y - 1, c, buffer[(x + y * width) * 4 + c]);
				}
			}
		}

		time_t currentTime = time(0);
		tm* currentDate = localtime(&currentTime);

		char out_name[1024];
#ifdef LIBPNG_SUPPORT
		sprintf(out_name, "snapshot-%d-%02d-%02d_%02d-%02d-%02d.png", currentDate->tm_year + 1900, currentDate->tm_mon + 1, currentDate->tm_mday, currentDate->tm_hour, currentDate->tm_min, currentDate->tm_sec);
#else
#ifdef LIBJPEG_SUPPORT
		sprintf(out_name, "snapshot-%d-%02d-%02d_%02d-%02d-%02d.jpg", currentDate->tm_year + 1900, currentDate->tm_mon + 1, currentDate->tm_mday, currentDate->tm_hour, currentDate->tm_min, currentDate->tm_sec);
#else
		sprintf(out_name, "snapshot-%d-%02d-%02d_%02d-%02d-%02d.pnm", currentDate->tm_year + 1900, currentDate->tm_mon + 1, currentDate->tm_mday, currentDate->tm_hour, currentDate->tm_min, currentDate->tm_sec);
#endif
#endif
		out.WriteImage(out_name);

		g_do_snapshot = false;
	}

	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
}

static bool first = true;
// use OpenGL to build view matrix
static GLfloat modelView[16];
static GLfloat modelRotateView[16];

// display results using OpenGL (called by GLUT)
void display(bool display)
{
	if (first)
	{
		sdkStartTimer(&timer);
		//first = false;
	}

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	if (first)
	{
		glLoadIdentity();
		first = false;
	}
	else
	{
		glLoadMatrixf(modelRotateView);
	}
	glRotatef(-viewRotation.x, 1.0, 0.0, 0.0);
	glRotatef(-viewRotation.y, 0.0, 1.0, 0.0);
	viewRotation.x = viewRotation.y = 0.0f;
	glGetFloatv(GL_MODELVIEW_MATRIX, modelRotateView);

	glTranslatef(-viewTranslation.x, -viewTranslation.y, -viewTranslation.z);
	glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
	glPopMatrix();

	viewMatrix[0] = modelView[0];
	viewMatrix[1] = modelView[1];
	viewMatrix[2] = modelView[2];
	viewMatrix[3] = modelView[3];

	viewMatrix[4] = modelView[4];
	viewMatrix[5] = modelView[5];
	viewMatrix[6] = modelView[6];
	viewMatrix[7] = modelView[7];

	viewMatrix[8] = modelView[8];
	viewMatrix[9] = modelView[9];
	viewMatrix[10] = modelView[10];
	viewMatrix[11] = modelView[11];

	invViewMatrix[0] = modelView[0];
	invViewMatrix[1] = modelView[4];
	invViewMatrix[2] = modelView[8];
	invViewMatrix[3] = modelView[12];

	invViewMatrix[4] = modelView[1];
	invViewMatrix[5] = modelView[5];
	invViewMatrix[6] = modelView[9];
	invViewMatrix[7] = modelView[13];

	invViewMatrix[8] = modelView[2];
	invViewMatrix[9] = modelView[6];
	invViewMatrix[10] = modelView[10];
	invViewMatrix[11] = modelView[14];

	if ((width > 1) && (height > 1))
	{
		float aspect = (width - 1.0f) / (height - 1.0f);
		if (aspect > 1.0f)
		{
			invViewMatrix[0] *= aspect;
			invViewMatrix[4] *= aspect;
			invViewMatrix[8] *= aspect;
		}
		else
		{
			invViewMatrix[1] /= aspect;
			invViewMatrix[5] /= aspect;
			invViewMatrix[7] /= aspect;
		}
	}

	//auto t0 = std::chrono::high_resolution_clock::now();
	render();
	if (!display) return;
	//auto t1 = std::chrono::high_resolution_clock::now();
	//std::chrono::nanoseconds ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0);
	//std::cout << ns.count() << " nanoseconds" << std::endl;

	// display results
	glClear(GL_COLOR_BUFFER_BIT);

	// draw image from PBO
	glDisable(GL_DEPTH_TEST);

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	// draw using texture

	// copy from pbo to texture
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	// draw textured quad
	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);
	glTexCoord2f(0, 0);
	glVertex2f(0, 0);
	glTexCoord2f(1, 0);
	glVertex2f(1, 0);
	glTexCoord2f(1, 1);
	glVertex2f(1, 1);
	glTexCoord2f(0, 1);
	glVertex2f(0, 1);
	glEnd();

	glDisable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, 0);

	glutSwapBuffers();
	glutReportErrors();

	//sdkStopTimer(&timer);

	computeFPS();
}

void display()
{
	display(true);
}

std::chrono::high_resolution_clock::time_point start, end;

void reshape(int w, int h);

void do_auto()
{
	if (auto_mode > 0)
	{
		unsigned int backup = auto_mode;
		srand(1234);
		unsigned int old_width = width;
		unsigned int old_height = height;
		auto_mode = 0;

		reshape(512, 512);
		first = true;
		auto_mode = backup;
		start = std::chrono::high_resolution_clock::now();
		while (auto_mode > 0)
		{
			viewRotation.x += 100.0f * rand() / (float)RAND_MAX - 50.0f;
			viewRotation.y += 100.0f * rand() / (float)RAND_MAX - 50.0f;
			display(false);
			auto_mode--;
		}
		end = std::chrono::high_resolution_clock::now();
		std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms = " << 250.0f * 1000.0f / (float)(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) << " fps" << std::endl;

		if (g_automode)
		{
			fprintf(g_logfile, "%d ms = %f fps\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(), 250.0f * 1000.0f / (float)(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()));
		}

		reshape(1920, 1080);
		first = true;
		auto_mode = backup;
		start = std::chrono::high_resolution_clock::now();
		while (auto_mode > 0)
		{
			viewRotation.x += 100.0f * rand() / (float)RAND_MAX - 50.0f;
			viewRotation.y += 100.0f * rand() / (float)RAND_MAX - 50.0f;
			display(false);
			auto_mode--;
		}
		end = std::chrono::high_resolution_clock::now();
		std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms = " << 250.0f * 1000.0f / (float)(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) << " fps" << std::endl;
		if (g_automode)
		{
			fprintf(g_logfile, "%d ms = %f fps\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(), 250.0f * 1000.0f / (float)(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()));
		}

		reshape(old_width, old_height);
	}
}

void idle()
{
	if (g_automode)
	{
		do_auto();
		fclose(g_logfile);
		exit(EXIT_SUCCESS);
	}
}

void keyboard(unsigned char key, int x, int y)
{
	int old_transfer = transfer;
	bool old_smooth_transfer = smooth_transfer;
	switch (key)
	{
		case 27:
			exit(EXIT_SUCCESS);
			break;

		case 'f':
			linearFiltering = !linearFiltering;
			if (volumeType == 0)
				renderer8->setTextureFilterMode(linearFiltering);
			else if (volumeType == 1)
				renderer16->setTextureFilterMode(linearFiltering);
			else if (volumeType == 2)
				renderer32->setTextureFilterMode(linearFiltering);
			else
				renderer64->setTextureFilterMode(linearFiltering);
			break;

		case 'b':
			backgroundWhite = !backgroundWhite;
			break;

		case 'c':
			rgbColor = !rgbColor;
			if (volumeType == 0)
				renderer8->enableRGBA(rgbColor);
			else if (volumeType == 1)
				renderer16->enableRGBA(rgbColor);
			else if (volumeType == 2)
				renderer32->enableRGBA(rgbColor);
			else
				renderer64->enableRGBA(rgbColor);
			break;

		case 'l':
			lighting = !lighting;
			if (volumeType == 0)
				renderer8->enableLighting(lighting);
			else if (volumeType == 1)
				renderer16->enableLighting(lighting);
			else if (volumeType == 2)
				renderer32->enableLighting(lighting);
			else
				renderer64->enableLighting(lighting);
			break;

		case 'a':
			auto_mode = 250;
			break;

		case '+':
			density += 0.01f;
			break;

		case '-':
			density -= 0.01f;
			break;

		case ']':
			brightness += 0.1f;
			break;

		case '[':
			brightness -= 0.1f;
			break;

		case ';':
			transferOffset += 0.01f;
			break;

		case '\'':
			transferOffset -= 0.01f;
			break;

		case '.':
			transferScale += 0.01f;
			break;

		case ',':
			transferScale -= 0.01f;
			break;

		case '0':
			transfer = transfer & 256;
			break;

		case '1':
			transfer = (transfer & 256) + 1;
			break;

		case '2':
			transfer = (transfer & 256) + 2;
			break;

		case '3':
			transfer = (transfer & 256) + 3;
			break;

		case '4':
			transfer = (transfer & 256) + 4;
			break;

		case '5':
			transfer = (transfer & 256) + 5;
			break;

		case '6':
			transfer = (transfer & 256) + 6;
			break;

		case '7':
			transfer = (transfer & 256) + 7;
			break;

		case '8':
			transfer = (transfer & 256) + 8;
			break;

		case '9':
			transfer = (transfer & 256) + 9;
			break;

		case 'q':
			std::cout << "Rotation:" << std::endl;
			for (int i = 0; i < 16; i++)
			{
				std::cout << " " << modelRotateView[i];
				if ((i & 3) == 3) std::cout << std::endl;
			}
			std::cout << "Translation:" << std::endl;
			std::cout << " " << viewTranslation.x << " " << viewTranslation.y << " " << viewTranslation.z << std::endl;
			break;
		case 'w':
			modelRotateView[ 0] = -0.23345f;
			modelRotateView[ 1] = 0.772606f;
			modelRotateView[ 2] = 0.590411f;
			modelRotateView[ 3] = 0.0f;
			modelRotateView[ 4] = 0.949943f;
			modelRotateView[ 5] = 0.0515598f;
			modelRotateView[ 6] = 0.308139f;
			modelRotateView[ 7] = 0.0f;
			modelRotateView[ 8] = 0.207629f;
			modelRotateView[ 9] = 0.632791f;
			modelRotateView[10] = -0.745967f;
			modelRotateView[11] = 0.0f;
			modelRotateView[12] = 0.0f;
			modelRotateView[13] = 0.0f;
			modelRotateView[14] = 0.0f;
			modelRotateView[15] = 1.0f;
			viewTranslation.x = 0.56f;
			viewTranslation.y = 0.28f;
			viewTranslation.z = -2.37f;
			break;
		case 'e':
			modelRotateView[0] = 0.251946f;
			modelRotateView[1] = 0.967615f;
			modelRotateView[2] = 0.0158761f;
			modelRotateView[3] = 0.0f;
			modelRotateView[4] = -0.3102f;
			modelRotateView[5] = 0.0652083f;
			modelRotateView[6] = 0.948435f;
			modelRotateView[7] = 0.0f;
			modelRotateView[8] = 0.916683f;
			modelRotateView[9] = -0.243881f;
			modelRotateView[10] = 0.316581f;
			modelRotateView[11] = 0.0f;
			modelRotateView[12] = 0.0f;
			modelRotateView[13] = 0.0f;
			modelRotateView[14] = 0.0f;
			modelRotateView[15] = 1.0f;
			viewTranslation.x = 0.06f;
			viewTranslation.y = 0.02f;
			viewTranslation.z = -2.35f;
			break;
		case 'r':
			modelRotateView[0] = 0.48534f;
			modelRotateView[1] = 0.847198f;
			modelRotateView[2] = 0.216107f;
			modelRotateView[3] = 0.0f;
			modelRotateView[4] = 0.873654f;
			modelRotateView[5] = -0.479616f;
			modelRotateView[6] = -0.081854f;
			modelRotateView[7] = 0.0f;
			modelRotateView[8] = 0.0343025f;
			modelRotateView[9] = 0.22853f;
			modelRotateView[10] = -0.972933f;
			modelRotateView[11] = 0.0f;
			modelRotateView[12] = 0.0f;
			modelRotateView[13] = 0.0f;
			modelRotateView[14] = 0.0f;
			modelRotateView[15] = 1.0f;
			viewTranslation.x = 0.09f;
			viewTranslation.y = 0.29f;
			viewTranslation.z = -2.46f;
			break;
		case 't':
			modelRotateView[0] = 0.914751f;
			modelRotateView[1] = -0.403029f;
			modelRotateView[2] = -0.0282682f;
			modelRotateView[3] = 0.0f;
			modelRotateView[4] = -0.124562f;
			modelRotateView[5] = -0.21478f;
			modelRotateView[6] = -0.968688f;
			modelRotateView[7] = 0.0f;
			modelRotateView[8] = 0.384337f;
			modelRotateView[9] = 0.889629f;
			modelRotateView[10] = -0.246671f;
			modelRotateView[11] = 0.0f;
			modelRotateView[12] = 0.0f;
			modelRotateView[13] = 0.0f;
			modelRotateView[14] = 0.0f;
			modelRotateView[15] = 1.0f;
			viewTranslation.x = -0.03f;
			viewTranslation.y = -1.42f;
			viewTranslation.z = -1.27f;
			break;
		case 'u':
			modelRotateView[0] = -0.998334f;
			modelRotateView[1] = -0.0016688f;
			modelRotateView[2] = 0.0576924f;
			modelRotateView[3] = 0.0f;
			modelRotateView[4] = -0.055271f;
			modelRotateView[5] = 0.315525f;
			modelRotateView[6] = -0.947307f;
			modelRotateView[7] = 0.0f;
			modelRotateView[8] = -0.0166227f;
			modelRotateView[9] = -0.948917f;
			modelRotateView[10] = -0.315092f;
			modelRotateView[11] = 0.0f;
			modelRotateView[12] = 0.0f;
			modelRotateView[13] = 0.0f;
			modelRotateView[14] = 0.0f;
			modelRotateView[15] = 1.0f;
			viewTranslation.x = -0.12f;
			viewTranslation.y = -0.06f;
			viewTranslation.z = -2.47f;
			break;
		case 'i':
			modelRotateView[0] = 0.654504f;
			modelRotateView[1] = -0.754952f;
			modelRotateView[2] = 0.0409014f;
			modelRotateView[3] = 0.0f;
			modelRotateView[4] = -0.201201f;
			modelRotateView[5] = -0.226068f;
			modelRotateView[6] = -0.953106f;
			modelRotateView[7] = 0.0f;
			modelRotateView[8] = 0.728796f;
			modelRotateView[9] = 0.615583f;
			modelRotateView[10] = -0.299859f;
			modelRotateView[11] = 0.0f;
			modelRotateView[12] = 0.0f;
			modelRotateView[13] = 0.0f;
			modelRotateView[14] = 0.0f;
			modelRotateView[15] = 1.0f;
			viewTranslation.x = -0.06f;
			viewTranslation.y = 0.2f;
			viewTranslation.z = -2.65f;
			break;
		case 'o':
			modelRotateView[0] = -0.799198f;
			modelRotateView[1] = 0.574619f;
			modelRotateView[2] = 0.176343f;
			modelRotateView[3] = 0.0f;
			modelRotateView[4] = 0.300931f;
			modelRotateView[5] = 0.636485f;
			modelRotateView[6] = -0.710162f;
			modelRotateView[7] = 0.0f;
			modelRotateView[8] = -0.520313f;
			modelRotateView[9] = -0.514493f;
			modelRotateView[10] = -0.681597f;
			modelRotateView[11] = 0.0f;
			modelRotateView[12] = 0.0f;
			modelRotateView[13] = 0.0f;
			modelRotateView[14] = 0.0f;
			modelRotateView[15] = 1.0f;
			viewTranslation.x = 0.09f;
			viewTranslation.y = 0.4f;
			viewTranslation.z = -2.42f;
			break;
		case 'p':
			modelRotateView[0] = 0.304312f;
			modelRotateView[1] = -0.945155f;
			modelRotateView[2] = -0.118651f;
			modelRotateView[3] = 0.0f;
			modelRotateView[4] = 0.47888f;
			modelRotateView[5] = 0.0441183f;
			modelRotateView[6] = 0.876773f;
			modelRotateView[7] = 0.0f;
			modelRotateView[8] = -0.82345f;
			modelRotateView[9] = -0.323632f;
			modelRotateView[10] = 0.466041f;
			modelRotateView[11] = 0.0f;
			modelRotateView[12] = 0.0f;
			modelRotateView[13] = 0.0f;
			modelRotateView[14] = 0.0f;
			modelRotateView[15] = 1.0f;
			viewTranslation.x = 0.0f;
			viewTranslation.y = -0.07f;
			viewTranslation.z = -2.53f;
			break;
		case 'k':
			modelRotateView[0] = 0.895919f;
			modelRotateView[1] = 0.0730129f;
			modelRotateView[2] = 0.438179f;
			modelRotateView[3] = 0.0f;
			modelRotateView[4] = -0.147572f;
			modelRotateView[5] = -0.881461f;
			modelRotateView[6] = 0.44861f;
			modelRotateView[7] = 0.0f;
			modelRotateView[8] = 0.418991f;
			modelRotateView[9] = -0.46658f;
			modelRotateView[10] = -0.778943f;
			modelRotateView[11] = 0.0f;
			modelRotateView[12] = 0.0f;
			modelRotateView[13] = 0.0f;
			modelRotateView[14] = 0.0f;
			modelRotateView[15] = 1.0f;
			viewTranslation.x = -0.02f;
			viewTranslation.y = 0.29f;
			viewTranslation.z = -2.38f;
			break;
		case 'j':
			modelRotateView[0] = -0.331459f;
			modelRotateView[1] = 0.921952f;
			modelRotateView[2] = 0.200364f;
			modelRotateView[3] = 0.0f;
			modelRotateView[4] = 0.658637f;
			modelRotateView[5] = 0.378169f;
			modelRotateView[6] = -0.650533f;
			modelRotateView[7] = 0.0f;
			modelRotateView[8] = -0.675529f;
			modelRotateView[9] = -0.083658f;
			modelRotateView[10] = -0.732577f;
			modelRotateView[11] = 0.0f;
			modelRotateView[12] = 0.0f;
			modelRotateView[13] = 0.0f;
			modelRotateView[14] = 0.0f;
			modelRotateView[15] = 1.0f;
			viewTranslation.x = 0.17f;
			viewTranslation.y = -0.36f;
			viewTranslation.z = -2.56f;
			break;
		case 's':
		{
			unsigned int old_width = width;
			unsigned int old_height = height;
			reshape(512, 512);
			g_do_snapshot = true;
			display(false);
			reshape(old_width, old_height);
			break;
		}
		case 'm':
		{
			unsigned int old_width = width;
			unsigned int old_height = height;
			reshape(1024, 1024);
			g_do_snapshot = true;
			display(false);
			reshape(old_width, old_height);
			break;
		}
		case 'x':
		{
			unsigned int old_width = width;
			unsigned int old_height = height;
			reshape(2048, 2048);
			g_do_snapshot = true;
			display(false);
			reshape(old_width, old_height);
			break;
		}
		case 'z':
		{
			unsigned int old_width = width;
			unsigned int old_height = height;
			reshape(4096, 4096);
			g_do_snapshot = true;
			display(false);
			reshape(old_width, old_height);
			break;
		}
		case '*':
		{
			smooth_transfer = !smooth_transfer;
			break;
		}
		default:
			break;
	}

	if ((old_transfer != transfer) || (old_smooth_transfer != smooth_transfer))
	{
		resetTF();
		if (volumeType == 0)
		{
			renderer8->updateTF();
		}
		else if (volumeType == 1)
		{
			renderer16->updateTF();
		}
		else if (volumeType == 2)
		{
			renderer32->updateTF();
		}
		else
		{
			renderer64->updateTF();
		}

	}

	printf("density = %.2f, brightness = %.2f, transferOffset = %.2f, transferScale = %.2f\n", density, brightness, transferOffset, transferScale);
	do_auto();
	glutPostRedisplay();
}

int ox, oy;
int buttonState = 0;

void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		buttonState  |= 1<<button;
	}
	else if (state == GLUT_UP)
	{
		buttonState = 0;
	}

	ox = x;
	oy = y;
	glutPostRedisplay();
}

void motion(int x, int y)
{
	float dx, dy;
	dx = (float)(x - ox);
	dy = (float)(y - oy);

	if (buttonState == 4)
	{
		// right = zoom
		viewTranslation.z += dy / 100.0f;
	}
	else if (buttonState == 2)
	{
		// middle = translate
		viewTranslation.x += dx / 100.0f;
		viewTranslation.y -= dy / 100.0f;
	}
	else if (buttonState == 1)
	{
		// left = rotate
		viewRotation.x += dy / 5.0f;
		viewRotation.y += dx / 5.0f;
	}

	ox = x;
	oy = y;
	glutPostRedisplay();
}

void reshape(int w, int h)
{
	width = w;
	height = h;
	initPixelBuffer();

	// calculate new grid size
	gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));

	glViewport(0, 0, w, h);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
	glutPostRedisplay();
}

void cleanup()
{
	sdkDeleteTimer(&timer);

	checkCudaErrors(cudaDeviceSynchronize());
	//renderer->freeCudaBuffers();
	if (volumeType == 0)
		delete renderer8;
	else if (volumeType == 1)
		delete renderer16;
	else if (volumeType == 2)
		delete renderer32;
	else
		delete renderer64;

	delete[] transferFunc;

	if (pbo)
	{
		cudaGraphicsUnregisterResource(cuda_pbo_resource);
		glDeleteBuffersARB(1, &pbo);
		glDeleteTextures(1, &tex);
	}
}

void initGL(int *argc, char **argv)
{
	// initialize GLUT callback functions
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutInitWindowSize(width, height);
	glutCreateWindow("CUDA volume rendering");

	glewInit();

	if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object"))
	{
		printf("Required OpenGL extensions missing.");
		exit(EXIT_SUCCESS);
	}
}

void initPixelBuffer()
{
	if (pbo)
	{
		// unregister this buffer object from CUDA C
		checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));

		// delete old buffer
		glDeleteBuffersARB(1, &pbo);
		glDeleteTextures(1, &tex);
	}

	// create pixel buffer object for display
	glGenBuffersARB(1, &pbo);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte)*4, 0, GL_STREAM_DRAW_ARB);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));

	// create texture for display
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);
}

// General initialization call for CUDA Device
int chooseCudaDevice(int argc, const char **argv, bool bUseOpenGL)
{
	int result = 0;

	if (bUseOpenGL)
	{
		result = findCudaGLDevice(argc, argv);
	}
	else
	{
		result = findCudaDevice(argc, argv);
	}

	return result;
}

void runSingleTest(const char *ref_file, const char *exec_path)
{
	bool bTestResult = true;

	uint *d_output;
	checkCudaErrors(cudaMalloc((void **)&d_output, width*height*sizeof(uint)));
	checkCudaErrors(cudaMemset(d_output, 0, width*height*sizeof(uint)));

	float modelView[16] =
	{
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 4.0f, 1.0f
	};

	invViewMatrix[0] = modelView[0];
	invViewMatrix[1] = modelView[4];
	invViewMatrix[2] = modelView[8];
	invViewMatrix[3] = modelView[12];
	invViewMatrix[4] = modelView[1];
	invViewMatrix[5] = modelView[5];
	invViewMatrix[6] = modelView[9];
	invViewMatrix[7] = modelView[13];
	invViewMatrix[8] = modelView[2];
	invViewMatrix[9] = modelView[6];
	invViewMatrix[10] = modelView[10];
	invViewMatrix[11] = modelView[14];

	// call CUDA kernel, writing results to PBO
	if (volumeType == 0)
		renderer8->copyInvViewMatrix(invViewMatrix, sizeof(float4) * 3);
	else if (volumeType == 1)
		renderer16->copyInvViewMatrix(invViewMatrix, sizeof(float4) * 3);
	else if (volumeType == 2)
		renderer32->copyInvViewMatrix(invViewMatrix, sizeof(float4) * 3);
	else
		renderer64->copyInvViewMatrix(invViewMatrix, sizeof(float4) * 3);

	// Start timer 0 and process n loops on the GPU
	int nIter = 10;

	for (int i = -1; i < nIter; i++)
	{
		if (i == 0)
		{
			checkCudaErrors(cudaDeviceSynchronize());
			sdkStartTimer(&timer);
		}

		float smallestExtent;
		if ((rgbColor) || (lighting))
			smallestExtent = std::min(std::min(g_scale.x / volumeSize.width, g_scale.y / volumeSize.height), g_scale.z / (volumeSize.depth / 4));
		else
			smallestExtent = std::min(std::min(g_scale.x / volumeSize.width, g_scale.y / volumeSize.height), g_scale.z / volumeSize.depth);

		tstep = std::min(0.01f, smallestExtent);
		dim3 warpDim;

		if (volumeType == 0)
		{
			renderer8->getLaunchParameter(warpDim, width, height, tstep);
			blockSize = dim3(warpDim.x * warpDim.z * 4, warpDim.y * 2);
			gridSize = dim3(iDivUp(width * warpDim.z, blockSize.x), iDivUp(height, blockSize.y));
			renderer8->render_kernel(gridSize, blockSize, warpDim, d_output, width, height, density, brightness, transferOffset, transferScale, tstep, backgroundWhite);
		}
		else if (volumeType == 1)
		{
			renderer16->getLaunchParameter(warpDim, width, height, tstep);
			blockSize = dim3(warpDim.x * warpDim.z * 4, warpDim.y * 2);
			gridSize = dim3(iDivUp(width * warpDim.z, blockSize.x), iDivUp(height, blockSize.y));
			renderer16->render_kernel(gridSize, blockSize, warpDim, d_output, width, height, density, brightness, transferOffset, transferScale, tstep, backgroundWhite);
		}
		else if (volumeType == 2)
		{
			renderer32->getLaunchParameter(warpDim, width, height, tstep);
			blockSize = dim3(warpDim.x * warpDim.z * 4, warpDim.y * 2);
			gridSize = dim3(iDivUp(width * warpDim.z, blockSize.x), iDivUp(height, blockSize.y));
			renderer32->render_kernel(gridSize, blockSize, warpDim, d_output, width, height, density, brightness, transferOffset, transferScale, tstep, backgroundWhite);
		}
		else
		{
			renderer64->getLaunchParameter(warpDim, width, height, tstep);
			blockSize = dim3(warpDim.x * warpDim.z * 4, warpDim.y * 2);
			gridSize = dim3(iDivUp(width * warpDim.z, blockSize.x), iDivUp(height, blockSize.y));
			renderer64->render_kernel(gridSize, blockSize, warpDim, d_output, width, height, density, brightness, transferOffset, transferScale, tstep, backgroundWhite);
		}

		//std::cout << "warpDim = " << warpDim.x << ", " << warpDim.y << ", " << warpDim.z << std::endl;
	}

	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&timer);
	// Get elapsed time and throughput, then log to sample and master logs
	double dAvgTime = sdkGetTimerValue(&timer)/(nIter * 1000.0);
	printf("volumeRender, Throughput = %.4f MTexels/s, Time = %.5f s, Size = %u Texels, NumDevsUsed = %u, Workgroup = %u\n",
		   (1.0e-6 * width * height)/dAvgTime, dAvgTime, (width * height), 1, blockSize.x * blockSize.y);


	getLastCudaError("Error: render_kernel() execution FAILED");
	checkCudaErrors(cudaDeviceSynchronize());

	unsigned char *h_output = (unsigned char *)malloc(width*height*4);
	checkCudaErrors(cudaMemcpy(h_output, d_output, width*height*4, cudaMemcpyDeviceToHost));

	sdkSavePPM4ub("volume.ppm", h_output, width, height);
	bTestResult = sdkComparePPM("volume.ppm", sdkFindFilePath(ref_file, exec_path), MAX_EPSILON_ERROR, THRESHOLD, true);

	checkCudaErrors(cudaFree(d_output));
	free(h_output);
	cleanup();

	exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

int log_map(int v)
{
	return (int)floor(256.0 * (1.0 - exp(-8.0 * log(2.0) * (double)v / 4095.0)) + 0.5);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
	pArgc = &argc;
	pArgv = argv;

	char *ref_file = NULL;

	//start logs
	printf("%s Starting...\n\n", sSDKsample);

	if (checkCmdLineFlag(argc, (const char **)argv, "file"))
	{
		getCmdLineArgumentString(argc, (const char **)argv, "file", &ref_file);
		fpsLimit = frameCheckNumber;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "swidth"))
	{
		width = getCmdLineArgumentInt(argc, (const char **)argv, "swidth");
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "sheight"))
	{
		height = getCmdLineArgumentInt(argc, (const char **)argv, "sheight");
	}

	// parse arguments
	char *filename;
	char *export_name = NULL;
	// default = latest
	int export_version = -1;

	if (getCmdLineArgumentString(argc, (const char **)argv, "volume", &filename))
	{
		volumeFilename = filename;
	}
	if (!(getCmdLineArgumentString(argc, (const char **)argv, "export", &export_name)))
	{
		export_name = NULL;
	}
	if (checkCmdLineFlag(argc, (const char **)argv, "export_version"))
	{
		export_version = getCmdLineArgumentInt(argc, (const char **)argv, "export_version");
	}

	char *compare_a = NULL;
	char *compare_b = NULL;
	
	if (getCmdLineArgumentString(argc, (const char **)argv, "compare_a", &filename))
	{
		compare_a = filename;
	}
	if (getCmdLineArgumentString(argc, (const char **)argv, "compare_b", &filename))
	{
		compare_b = filename;
	}

	if ((compare_a != NULL) && (compare_b != NULL))
	{
		FILE *in_a = fopen(compare_a, "rb");
		FILE *in_b = fopen(compare_b, "rb");

		uint32_t header_a[16];
		uint32_t header_b[16];
		fread(header_a, sizeof(uint32_t), 16, in_a);
		fread(header_b, sizeof(uint32_t), 16, in_b);
		char dummy;
		for (unsigned int i = 0; i < header_a[15]; i++) fread(&dummy, 1, 1, in_a);
		for (unsigned int i = 0; i < header_b[15]; i++) fread(&dummy, 1, 1, in_b);
		uint32_t size_a, size_b;
		fread(&size_a, sizeof(uint32_t), 1, in_a);
		fread(&size_b, sizeof(uint32_t), 1, in_b);
		if (size_a != size_b)
		{
			std::cout << "different size" << std::endl;
			exit(-1);
		}
		double psnr = 0.0;
		double peak;
		if (header_a[4] == GL_UNSIGNED_BYTE)
		{
			peak = 255.0;
			unsigned char a, b;
			for (unsigned int i = 0; i < size_a; i++)
			{
				fread(&a, sizeof(unsigned char), 1, in_a);
				fread(&b, sizeof(unsigned char), 1, in_b);
				double dlt = (double)a - (double)b;
				psnr += dlt * dlt;
			}
		}
		else
		{
			peak = 4095.0;
			size_a >>= 1;
			size_b >>= 1;
			unsigned short a, b;
			for (unsigned int i = 0; i < size_a; i++)
			{
				fread(&a, sizeof(unsigned short), 1, in_a);
				fread(&b, sizeof(unsigned short), 1, in_b);
				double dlt = (double)a - (double)b;
				psnr += dlt * dlt;
			}
		}
		psnr /= (double) size_a;
		psnr = peak * peak / psnr;
		psnr = 10.0 * log(psnr) / log(10.0);
		std::cout << "  PSNR: " << psnr << "db" << std::endl;
		exit(0);
	}

	int n;
	float3 scale;
	scale.x = scale.y = scale.z = 1.0f;
	int raw = 0;
	int compression = 1;
	int linear = 0;
	int denoise = 0;
	int lossy = 0;
	int max_error = 0;
	bool png = false;
	int start = 0;
	int end = 0;
	int clip_x0, clip_x1, clip_y0, clip_y1;
	float scale_png = 1.0f;
	bool histogram = false;
	//int transfer = 0;
	int raw_skip = 0;
	bool calc_gradient = false;
	bool clip_zero = false;
	int raw_components = 1;
	bool compare = false;
	displayEntropy = false;
	bool empty = false;
	cudaExtent resampleSize;
	resampleSize.width = resampleSize.height = resampleSize.depth = 0;
	bool expand = false;
	bool bruteForce= false;

	volumeType = 0;

	if (checkCmdLineFlag(argc, (const char **)argv, "size"))
	{
		n = getCmdLineArgumentInt(argc, (const char **)argv, "size");
		volumeSize.width = volumeSize.height = volumeSize.depth = n;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "empty"))
	{
		n = getCmdLineArgumentInt(argc, (const char **)argv, "empty");
		volumeSize.width = volumeSize.height = volumeSize.depth = n;
		empty = true;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "xsize"))
	{
		n = getCmdLineArgumentInt(argc, (const char **)argv, "xsize");
		volumeSize.width = n;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "ysize"))
	{
		n = getCmdLineArgumentInt(argc, (const char **)argv, "ysize");
		volumeSize.height = n;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "zsize"))
	{
		n = getCmdLineArgumentInt(argc, (const char **)argv, "zsize");
		volumeSize.depth = n;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "xoriginal"))
	{
		n = getCmdLineArgumentInt(argc, (const char **)argv, "xoriginal");
		originalSize.width = n;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "yoriginal"))
	{
		n = getCmdLineArgumentInt(argc, (const char **)argv, "yoriginal");
		originalSize.height = n;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "zoriginal"))
	{
		n = getCmdLineArgumentInt(argc, (const char **)argv, "zoriginal");
		originalSize.depth = n;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "xresample"))
	{
		n = getCmdLineArgumentInt(argc, (const char **)argv, "xresample");
		resampleSize.width = n;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "yresample"))
	{
		n = getCmdLineArgumentInt(argc, (const char **)argv, "yresample");
		resampleSize.height = n;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "zresample"))
	{
		n = getCmdLineArgumentInt(argc, (const char **)argv, "zresample");
		resampleSize.depth = n;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "expand"))
	{
		expand = true;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "xscale"))
	{
		scale.x = getCmdLineArgumentFloat(argc, (const char **)argv, "xscale");
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "yscale"))
	{
		scale.y = getCmdLineArgumentFloat(argc, (const char **)argv, "yscale");
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "zscale"))
	{
		scale.z = getCmdLineArgumentFloat(argc, (const char **)argv, "zscale");
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "histogram"))
	{
		histogram = true;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "raw"))
	{
		n = getCmdLineArgumentInt(argc, (const char **)argv, "raw");
		raw = n;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "rgba"))
	{
		volumeType |= 2;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "combined"))
	{
		volumeType |= 2;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "split"))
	{
		volumeType &= ~2;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "png"))
	{
		png = true;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "basic"))
	{
		basic = true;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "learning"))
	{
		learning = true;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "fixed"))
	{
		fixed = true;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "fast"))
	{
		g_fast = true;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "scale_png"))
	{
		scale_png = getCmdLineArgumentFloat(argc, (const char **)argv, "scale_png");
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "raw_skip"))
	{
		n = getCmdLineArgumentInt(argc, (const char **)argv, "raw_skip");
		raw_skip = n;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "raw_components"))
	{
		n = getCmdLineArgumentInt(argc, (const char **)argv, "raw_components");
		raw_components = n;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "start"))
	{
		n = getCmdLineArgumentInt(argc, (const char **)argv, "start");
		start = n;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "end"))
	{
		n = getCmdLineArgumentInt(argc, (const char **)argv, "end");
		end = n;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "clip_x0"))
	{
		n = getCmdLineArgumentInt(argc, (const char **)argv, "clip_x0");
		clip_x0 = n;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "clip_x1"))
	{
		n = getCmdLineArgumentInt(argc, (const char **)argv, "clip_x1");
		clip_x1 = n;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "clip_y0"))
	{
		n = getCmdLineArgumentInt(argc, (const char **)argv, "clip_y0");
		clip_y0 = n;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "clip_y1"))
	{
		n = getCmdLineArgumentInt(argc, (const char **)argv, "clip_y1");
		clip_y1 = n;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "16bit"))
	{
		volumeType |= 1;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "8bit"))
	{
		volumeType &= ~1;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "clip_zero"))
	{
		clip_zero = true;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "gradient"))
	{
		calc_gradient = true;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "compare"))
	{
		compare = true;
		displayEntropy = true;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "linear"))
	{
		n = getCmdLineArgumentInt(argc, (const char **)argv, "linear");
		linear = n;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "denoise"))
	{
		n = getCmdLineArgumentInt(argc, (const char **)argv, "denoise");
		denoise = n;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "lossy"))
	{
		n = getCmdLineArgumentInt(argc, (const char **)argv, "lossy");
		lossy = n;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "max_error"))
	{
		n = getCmdLineArgumentInt(argc, (const char **)argv, "max_error");
		max_error = n;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "compression"))
	{
		n = getCmdLineArgumentInt(argc, (const char **)argv, "compression");
		compression = n;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "auto"))
	{
		g_automode = true;
		auto_mode = 250;
		g_logfile = fopen("logfile.txt", "a");
		for (int i = 0; i < argc; i++)
		{
			fprintf(g_logfile, "%s", argv[i]);
			if (i + 1 < argc) fprintf(g_logfile, " ");
			else fprintf(g_logfile, "\n");
		}
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "lighting"))
	{
		lighting = true;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "color"))
	{
		rgbColor = true;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "transfer"))
	{
		n = getCmdLineArgumentInt(argc, (const char **)argv, "transfer");
		transfer = n;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "density"))
	{
		density = getCmdLineArgumentFloat(argc, (const char **)argv, "density");
	}
	
	if (checkCmdLineFlag(argc, (const char **)argv, "brightness"))
	{
		brightness = getCmdLineArgumentFloat(argc, (const char **)argv, "brightness");
	}
	
	if (checkCmdLineFlag(argc, (const char **)argv, "transferScale"))
	{
		transferScale = getCmdLineArgumentFloat(argc, (const char **)argv, "transferScale");
	}
	
	if (checkCmdLineFlag(argc, (const char **)argv, "transferOffset"))
	{
		transferOffset = getCmdLineArgumentFloat(argc, (const char **)argv, "transferOffset");
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "bruteForce"))
	{
		bruteForce = true;
	}

	// load volume data
	char *path = sdkFindFilePath(volumeFilename, argv[0]);

	if (path == 0)
	{
		size_t len = strlen(volumeFilename);
		path = new char[len + 1];
		strcpy(path, volumeFilename);
		//		printf("Error finding file '%s'\n", volumeFilename);
		//		exit(EXIT_FAILURE);
	}

	char *nameWithoutPath;
	getFileNameWithoutPath(path, &nameWithoutPath);
	printf("Reading volume data file %s\n\n", nameWithoutPath);

	switch (compression)
	{
	case 0:
		if (volumeType == 0)
			renderer8 = new BasicRenderer<VolumeType8>();
		else if (volumeType == 1)
			renderer16 = new BasicRenderer<VolumeType16>();
		else if (volumeType == 2)
			renderer32 = new BasicRenderer<VolumeType32>();
		else
			renderer64 = new BasicRenderer<VolumeType64>();
		break;
	case 1:
		if (volumeType == 0)
			renderer8 = new RendererRBUC8x8<VolumeType8>();
		else if (volumeType == 1)
			renderer16 = new RendererRBUC8x8<VolumeType16>();
		else if (volumeType == 2)
			renderer32 = new RendererRBUC8x8<VolumeType32>();
		else
			renderer64 = new RendererRBUC8x8<VolumeType64>();
		break;
	default:
		std::cerr << "Illegal compression value." << std::endl;
		exit(-1);
	}

	size_t size;
	std::vector<size_t> hist;

	void *raw_volume;
	unsigned int element_size = 0;
	unsigned int element_count = 0;
	if (empty)
	{
		size = volumeSize.width * volumeSize.height * volumeSize.depth * sizeof(VolumeType16);
		raw_volume = malloc(size);
		memset(raw_volume, 0, size);
		element_size = 2;
		element_count = 1;
	}
	else if (raw == 1)
	{
		size = volumeSize.width * volumeSize.height * volumeSize.depth * sizeof(VolumeType8);
		raw_volume = loadRawFile<VolumeType8>(path, size, scale, raw_skip);
		element_size = 1;
		element_count = raw_components;
	}
	else if (raw == 2)
	{
		size = volumeSize.width * volumeSize.height * volumeSize.depth * sizeof(VolumeType16);
		raw_volume = loadRawFile<VolumeType16>(path, size, scale, raw_skip);
		element_size = 2;
		element_count = raw_components;
	}
#ifdef LIBPNG_SUPPORT
	else if (png)
	{
		element_count = getPngComponents(path, start);
		if ((element_size = getPngElementSize(path, start)) == 1)
		{
			raw_volume = loadPngFiles<VolumeType8>(path, volumeSize, scale, start, end, clip_x0, clip_x1, clip_y0, clip_y1, scale_png, clip_zero);
		}
		else
		{
			raw_volume = loadPngFiles<VolumeType16>(path, volumeSize, scale, start, end, clip_x0, clip_x1, clip_y0, clip_y1, scale_png, clip_zero);
		}
	}
#endif
	else
	{
		raw_volume = loadDatFile(path, volumeSize, scale, element_size, element_count);
	}

	size = volumeSize.width * volumeSize.height * volumeSize.depth * element_size;
	volumeSize.depth /= element_count;

	if (resampleSize.width == 0) resampleSize.width = volumeSize.width;
	if (resampleSize.height == 0) resampleSize.height = volumeSize.height;
	if (resampleSize.depth == 0) resampleSize.depth = volumeSize.depth;

	if ((resampleSize.width != volumeSize.width) || (resampleSize.height != volumeSize.height) || (resampleSize.depth != volumeSize.depth))
	{
		void *tmp_volume = malloc(element_size * element_count * resampleSize.width * resampleSize.height * resampleSize.depth);
		if (element_size == 1)
		{
			for (unsigned int c = 0; c < element_count; c++)
			{
				resampleVolume<VolumeType8>(&(((VolumeType8 *)raw_volume)[c * volumeSize.width * volumeSize.height * volumeSize.depth]), &(((VolumeType8 *)tmp_volume)[c * resampleSize.width * resampleSize.height * resampleSize.depth]), volumeSize, resampleSize);
			}
		}
		else
		{
			for (unsigned int c = 0; c < element_count; c++)
			{
				resampleVolume<VolumeType16>(&(((VolumeType16 *)raw_volume)[c * volumeSize.width * volumeSize.height * volumeSize.depth]), &(((VolumeType16 *)tmp_volume)[c * resampleSize.width * resampleSize.height * resampleSize.depth]), volumeSize, resampleSize);
			}
		}
		std::swap(raw_volume, tmp_volume);
		free(tmp_volume);
		volumeSize = resampleSize;
	}

	if (expand)
	{
		resampleSize.width = (volumeSize.width + 3ll) & (~3ll);
		resampleSize.height = (volumeSize.height + 3ll) & (~3ll);
		resampleSize.depth = (volumeSize.depth + 3ll) & (~3ll);
		if ((resampleSize.width != volumeSize.width) || (resampleSize.height != volumeSize.height) || (resampleSize.depth != volumeSize.depth))
		{
			void *tmp_volume = malloc(element_size * element_count * resampleSize.width * resampleSize.height * resampleSize.depth);
			if (element_size == 1)
			{
				for (unsigned int c = 0; c < element_count; c++)
				{
					expandVolume<VolumeType8>(&(((VolumeType8 *)raw_volume)[c * volumeSize.width * volumeSize.height * volumeSize.depth]), &(((VolumeType8 *)tmp_volume)[c * volumeSize.width * volumeSize.height * volumeSize.depth]), volumeSize, resampleSize);
				}
			}
			else
			{
				for (unsigned int c = 0; c < element_count; c++)
				{
					expandVolume<VolumeType16>(&(((VolumeType16 *)raw_volume)[c * volumeSize.width * volumeSize.height * volumeSize.depth]), &(((VolumeType16 *)tmp_volume)[c * volumeSize.width * volumeSize.height * volumeSize.depth]), volumeSize, resampleSize);
				}
			}
			std::swap(raw_volume, tmp_volume);
			free(tmp_volume);
			volumeSize = resampleSize;
		}
	}

	if (denoise != 0)
	{
		if (element_size == 1)
		{
			for (unsigned int c = 0; c < element_count; c++)
			{
				denoiseVolume<VolumeType8>(&(((VolumeType8 *)raw_volume)[c * volumeSize.width * volumeSize.height * volumeSize.depth]), volumeSize, denoise);
			}
		}
		else
		{
			for (unsigned int c = 0; c < element_count; c++)
			{
				denoiseVolume<VolumeType16>(&(((VolumeType16 *)raw_volume)[c * volumeSize.width * volumeSize.height * volumeSize.depth]), volumeSize, denoise);
			}
		}
	}

	// conversion
	if ((element_size == 2) && ((volumeType == 0) || (volumeType == 2)))
	{
		void *tmp = malloc(element_count * volumeSize.width * volumeSize.height * volumeSize.depth);

		if (linear != 0)
		{
			int max_val = 0;
			for (size_t i = 0; i < volumeSize.width * volumeSize.height * volumeSize.depth * element_count; i++) max_val = std::max(max_val, (int)((unsigned short *)raw_volume)[i]);
			std::cout << "maximum value: " << max_val << std::endl;
			float f = logf(255.0f) / logf((float)max_val);
			for (size_t i = 0; i < volumeSize.width * volumeSize.height * volumeSize.depth * element_count; i++)
			{
				if (linear == 1)
				{
					// linear mapping
					float v = ((float)((unsigned short *)raw_volume)[i]) / ((float)max_val);
					((unsigned char *)tmp)[i] = (unsigned char)floor(v * 255.0f + 0.5f);
				}
				else if (linear == 2)
				{
					// polynomial mapping
					if (((unsigned short *)raw_volume)[i] == 0)
						((unsigned char *)tmp)[i] = 0;
					else
						((unsigned char *)tmp)[i] = (unsigned char)floor(expf(logf((float)((unsigned short *)raw_volume)[i]) * f) + 0.5f);
				}
				else
				{
					((unsigned char *)tmp)[i] = (unsigned char)log_map(((unsigned short *)raw_volume)[i]);
				}
			}
			// adapt transfer function
			transferFuncMappingActive = true;
			int idx = 0;

			for (int i = 0; i < 256; i++)
			{
				if (linear == 1)
				{
					idx = (int)floor((float)i * (float)max_val / 255.0f + 0.5f);
				}
				else if (linear == 2)
				{
					if (i == 0)
						idx = 0;
					else
						idx = (int)floor(expf(logf((float)i) / f) + 0.5f);
				}
				else
				{
					while (log_map(idx) < i) idx++;
				}
				transferFuncMapping[i] = idx;
			}
		}
		else
		{
			std::vector<int> mapping(65536);
			{
				std::vector<float> count(65536);
				for (int i = 0; i < 65536; i++) count[i] = 0.0f;
				for (size_t i = 0; i < volumeSize.width * volumeSize.height * volumeSize.depth * element_count; i++)
				{
					count[((unsigned short *)raw_volume)[i]] += 1.0f;
				}
				int unique = 0;

				for (int i = 0; i < 65536; i++)
				{
					mapping[i] = unique;
					if (count[i] > 0.0f) unique++;
				}
				while (unique > 256)
				{
					float min_merge = 0.0f;
					int merge_idx = -1;
					int start = 0;
					for (int i = 0; i < unique - 1; i++)
					{
						float cur_merge = 0.0f;
						for (int j = start; j < 65536; j++)
						{
							if (mapping[j] == i)
							{
								cur_merge += count[j];
								start++;
							}
							else if (mapping[j] == i + 1)
							{
								cur_merge += count[j];
							}
							else
							{
								break;
							}
						}
						if ((merge_idx == -1) || (cur_merge < min_merge))
						{
							min_merge = cur_merge;
							merge_idx = i;
						}
					}
					for (int i = 0; i < 65536; i++)
					{
						if (mapping[i] > merge_idx) mapping[i]--;
					}
					unique--;
				}
			}
			for (size_t i = 0; i < volumeSize.width * volumeSize.height * volumeSize.depth; i++)
			{
				((unsigned char *)tmp)[i] = (unsigned char)mapping[((unsigned short *)raw_volume)[i]];
			}
			// adapt transfer function
			transferFuncMappingActive = true;
			int idx = 0;

			for (int i = 0; i < 256; i++)
			{
				while (mapping[idx] < i) idx++;
				transferFuncMapping[i] = idx;
			}
		}

		std::swap(raw_volume, tmp);
		free(tmp);
		element_size = 1;
	}

	resetTF();

	if (calc_gradient)
	{
		if (element_size == 1)
		{
			raw_volume = (void *)calculateGradients<VolumeType8>((VolumeType8 *)raw_volume, volumeSize, element_count);
		}
		else
		{
			raw_volume = (void *)calculateGradients<VolumeType16>((VolumeType16 *)raw_volume, volumeSize, element_count);
		}
	}

	if (element_size == 1)
	{
		if ((volumeType == 1) || (volumeType == 3))
		{
			printf("Error file '%s' only contains 8 bit data\n", filename);
			exit(-1);
		}
		if (volumeType == 2)
		{
			if (element_count < 4)
			{
				printf("Error file '%s' only contains %d components\n", filename, element_count);
				exit(-1);
			}
			void *tmp = malloc(4 * volumeSize.width * volumeSize.height * volumeSize.depth);
#pragma omp parallel for
			for (int z = 0; z < volumeSize.depth; z++)
				for (unsigned int y = 0; y < volumeSize.height; y++)
					for (unsigned int x = 0; x < volumeSize.width; x++)
					{
						size_t i = x + (y + z * volumeSize.height) * volumeSize.width;
						((uchar4 *)tmp)[i].x = ((uchar *)raw_volume)[i];
						((uchar4 *)tmp)[i].y = ((uchar *)raw_volume)[i + volumeSize.width * volumeSize.height * volumeSize.depth];
						((uchar4 *)tmp)[i].z = ((uchar *)raw_volume)[i + volumeSize.width * volumeSize.height * volumeSize.depth * 2];
						((uchar4 *)tmp)[i].w = ((uchar *)raw_volume)[i + volumeSize.width * volumeSize.height * volumeSize.depth * 3];
					}
			std::swap(raw_volume, tmp);
			free(tmp);
			element_count = 1;
		}
	}
	else
	{
		if ((volumeType == 0) || (volumeType == 2))
		{
			printf("Data should be 8 bit already\n");
			exit(-1);
		}
		if (volumeType == 3)
		{
			if (element_count < 4)
			{
				printf("Error file '%s' only contains %d components\n", filename, element_count);
				exit(-1);
			}
			void *tmp = malloc(4 * sizeof(unsigned short) * volumeSize.width * volumeSize.height * volumeSize.depth);
#pragma omp parallel for
			for (int z = 0; z < volumeSize.depth; z++)
				for (unsigned int y = 0; y < volumeSize.height; y++)
					for (unsigned int x = 0; x < volumeSize.width; x++)
					{
						size_t i = x + (y + z * volumeSize.height) * volumeSize.width;
						((ushort4 *)tmp)[i].x = ((ushort *)raw_volume)[i];
						((ushort4 *)tmp)[i].y = ((ushort *)raw_volume)[i + volumeSize.width * volumeSize.height * volumeSize.depth];
						((ushort4 *)tmp)[i].z = ((ushort *)raw_volume)[i + volumeSize.width * volumeSize.height * volumeSize.depth * 2];
						((ushort4 *)tmp)[i].w = ((ushort *)raw_volume)[i + volumeSize.width * volumeSize.height * volumeSize.depth * 3];
					}
			std::swap(raw_volume, tmp);
			free(tmp);
			element_count = 1;
		}
	}

	if (lossy != 0)
	{
		switch (volumeType)
		{
		case 0:
			for (unsigned int c = 0; c < element_count; c++)
			{
				quantizeVolume<VolumeType8, unsigned long long, uint, uchar>(&(((VolumeType8 *)raw_volume)[c * volumeSize.width * volumeSize.height * volumeSize.depth]), volumeSize, lossy, bruteForce);
			}
			break;
		case 1:
			for (unsigned int c = 0; c < element_count; c++)
			{
				quantizeVolume<VolumeType16, unsigned long long, unsigned long long, ushort>(&(((VolumeType16 *)raw_volume)[c * volumeSize.width * volumeSize.height * volumeSize.depth]), volumeSize, lossy, bruteForce);
			}
			break;
		case 2:
			quantizeVolume<VolumeType32, unsigned long long, uint4, uchar4>((VolumeType32 *)raw_volume, volumeSize, lossy, bruteForce);
			break;
		default:
			quantizeVolume<VolumeType64, unsigned long long, ulonglong4, ushort4>((VolumeType64 *)raw_volume, volumeSize, lossy, bruteForce);
		}
	}

	if (compare)
	{
		switch (volumeType)
		{
		case 0:
			printSize<VolumeType8>((VolumeType8 *)raw_volume, volumeSize, element_count);
			break;
		case 1:
			printSize<VolumeType16>((VolumeType16 *)raw_volume, volumeSize, element_count);
			break;
		case 2:
			printSize<VolumeType32>((VolumeType32 *)raw_volume, volumeSize, element_count);
			break;
		default:
			printSize<VolumeType64>((VolumeType64 *)raw_volume, volumeSize, element_count);
		}
	}

	if (export_name != NULL)
	{
		saveDatFile(export_name, volumeSize, scale, element_size, element_count, raw_volume, volumeType, export_version);
	}

	printf("Volume size: %dx%dx%d %dbit\n", int(volumeSize.width), int(volumeSize.height), int(volumeSize.depth), element_size * element_count * 8);
	printf("Voxel extent: %fx%fx%f\n", scale.x, scale.y, scale.z);

	if (histogram)
	{
		// dump histogram image
#ifdef LIBPNG_SUPPORT
		PngImage out;
		out.SetBitDepth(8);
#else
#ifdef LIBJPEG_SUPPORT
		JpegImage out;
		out.SetQuality(100);
#else
		PnmImage out;
		out.SetBitDepth(8);
#endif
#endif

		unsigned int hist_width;
		unsigned int hist_height = 512;

		if (element_size == 1) hist_width = 256;
		else hist_width = 4096;

		out.SetWidth(hist_width);
		out.SetHeight(hist_height);
		out.SetComponents(1);

		if ((element_count > 1) || (volumeType > 1))
		{
			// 2D, alpha is fourth component
			double max_square = 0.0;
			switch (volumeType)
			{
			case 0:
				{
					size_t vol_size = volumeSize.width * volumeSize.height * volumeSize.depth;
					for (size_t i = 0; i < volumeSize.width * volumeSize.height * volumeSize.depth; i++)
					{
						double x = (double)(((unsigned char *)raw_volume)[i]) - 128.0;
						double y = (double)(((unsigned char *)raw_volume)[i + vol_size]) - 128.0;
						double z = (double)(((unsigned char *)raw_volume)[i + 2 * vol_size]) - 128.0;
						double square = x * x + y * y + z * z;
						max_square = std::max(max_square, square);
					}
				}
				break;
			case 1:
				{
					size_t vol_size = volumeSize.width * volumeSize.height * volumeSize.depth;
					for (size_t i = 0; i < volumeSize.width * volumeSize.height * volumeSize.depth; i++)
					{
						double x = (double)(((unsigned short *)raw_volume)[i]) - 2048.0;
						double y = (double)(((unsigned short *)raw_volume)[i + vol_size]) - 2048.0;
						double z = (double)(((unsigned short *)raw_volume)[i + 2 * vol_size]) - 2048.0;
						double square = x * x + y * y + z * z;
						max_square = std::max(max_square, square);
					}
				}
				break;
			case 2:
				{
					size_t vol_size = volumeSize.width * volumeSize.height * volumeSize.depth;
					for (size_t i = 0; i < volumeSize.width * volumeSize.height * volumeSize.depth; i++)
					{
						double x = (double)(((unsigned char *)raw_volume)[i * 4]) - 128.0;
						double y = (double)(((unsigned char *)raw_volume)[i * 4 + 1]) - 128.0;
						double z = (double)(((unsigned char *)raw_volume)[i * 4 + 2]) - 128.0;
						double square = x * x + y * y + z * z;
						max_square = std::max(max_square, square);
					}
				}
				break;
			default:
				{
					size_t vol_size = volumeSize.width * volumeSize.height * volumeSize.depth;
					for (size_t i = 0; i < volumeSize.width * volumeSize.height * volumeSize.depth; i++)
					{
						double x = (double)(((unsigned short *)raw_volume)[i * 4]) - 2048.0;
						double y = (double)(((unsigned short *)raw_volume)[i * 4 + 1]) - 2048.0;
						double z = (double)(((unsigned short *)raw_volume)[i * 4 + 2]) - 2048.0;
						double square = x * x + y * y + z * z;
						max_square = std::max(max_square, square);
					}
				}
			}
			double max_len = sqrt(max_square);
			size_t *count = new size_t[hist_width * hist_height];
			for (unsigned int i = 0; i < hist_width * hist_height; i++) count[i] = 0;
			switch (volumeType)
			{
			case 0:
				{
					size_t vol_size = volumeSize.width * volumeSize.height * volumeSize.depth;
					for (size_t i = 0; i < volumeSize.width * volumeSize.height * volumeSize.depth; i++)
					{
						double x = (double)(((unsigned char *)raw_volume)[i]) - 128.0;
						double y = (double)(((unsigned char *)raw_volume)[i + vol_size]) - 128.0;
						double z = (double)(((unsigned char *)raw_volume)[i + 2 * vol_size]) - 128.0;
						unsigned int w = ((unsigned char *)raw_volume)[i + 3 * vol_size];
						unsigned int len = (unsigned int)(floor((hist_height - 1.0) * sqrt(x * x + y * y + z * z) / max_len + 0.5));
						count[w + len * hist_width]++;
					}
				}
				break;
			case 1:
				{
					size_t vol_size = volumeSize.width * volumeSize.height * volumeSize.depth;
					for (size_t i = 0; i < volumeSize.width * volumeSize.height * volumeSize.depth; i++)
					{
						double x = (double)(((unsigned short *)raw_volume)[i]) - 2048.0;
						double y = (double)(((unsigned short *)raw_volume)[i + vol_size]) - 2048.0;
						double z = (double)(((unsigned short *)raw_volume)[i + 2 * vol_size]) - 2048.0;
						unsigned int w = ((unsigned short *)raw_volume)[i + 3 * vol_size];
						unsigned int len = (unsigned int)(floor((hist_height - 1.0) * sqrt(x * x + y * y + z * z) / max_len + 0.5));
						count[w + len * hist_width]++;
					}
				}
				break;
			case 2:
				{
					size_t vol_size = volumeSize.width * volumeSize.height * volumeSize.depth;
					for (size_t i = 0; i < volumeSize.width * volumeSize.height * volumeSize.depth; i++)
					{
						double x = (double)(((unsigned char *)raw_volume)[i * 4]) - 128.0;
						double y = (double)(((unsigned char *)raw_volume)[i * 4 + 1]) - 128.0;
						double z = (double)(((unsigned char *)raw_volume)[i * 4 + 2]) - 128.0;
						unsigned int w = ((unsigned char *)raw_volume)[i * 4 + 3];
						unsigned int len = (unsigned int)(floor((hist_height - 1.0) * sqrt(x * x + y * y + z * z) / max_len + 0.5));
						count[w + len * hist_width]++;
					}
				}
				break;
			default:
				{
					size_t vol_size = volumeSize.width * volumeSize.height * volumeSize.depth;
					for (size_t i = 0; i < volumeSize.width * volumeSize.height * volumeSize.depth; i++)
					{
						double x = (double)(((unsigned short *)raw_volume)[i * 4]) - 2048.0;
						double y = (double)(((unsigned short *)raw_volume)[i * 4 + 1]) - 2048.0;
						double z = (double)(((unsigned short *)raw_volume)[i * 4 + 2]) - 2048.0;
						unsigned int w = ((unsigned short *)raw_volume)[i * 4 + 3];
						unsigned int len = (unsigned int)(floor((hist_height - 1.0) * sqrt(x * x + y * y + z * z) / max_len + 0.5));
						count[w + len * hist_width]++;
					}
				}
			}

			size_t max_count = 0;
			for (unsigned int i = hist_height; i < hist_width * hist_height; i++) max_count = std::max(max_count, count[i]);

			for (unsigned int x = 0; x < hist_width; x++)
			{
				for (unsigned int y = 0; y < hist_height; y++)
				{
					double rel = std::min(1.0, (double)count[x + y * hist_width] / (double)max_count);
					unsigned int val = (unsigned int)floor(hist_height * sqrt(sqrt(rel)) + 0.5);
					out.VSetValue(x, hist_height - y - 1, 0, val);
				}
			}

			delete[] count;
		}
		else
		{
			// 1D
			size_t *count = new size_t[hist_width];
			for (unsigned int i = 0; i < hist_width; i++) count[i] = 0;

			if (element_size == 1)
			{
				for (size_t i = 0; i < volumeSize.width * volumeSize.height * volumeSize.depth; i++)
					count[((unsigned char *)raw_volume)[i]]++;
			}
			else
			{
				for (size_t i = 0; i < volumeSize.width * volumeSize.height * volumeSize.depth; i++)
					count[((unsigned short *)raw_volume)[i]]++;
			}

			size_t max_count = 0;
			for (unsigned int i = 1; i < hist_width; i++) max_count = std::max(max_count, count[i]);

			for (unsigned int x = 0; x < hist_width; x++)
			{
				double rel = std::min(1.0, (double)count[x] / (double)max_count);
				unsigned int val = (unsigned int)floor(hist_height * sqrt(rel) + 0.5);
				for (unsigned int y = 0; y < hist_height; y++)
				{
					out.VSetValue(x, hist_height - y - 1, 0, (y < val)?255:0);
				}
			}

			delete[] count;
		}

		time_t currentTime = time(0);
		tm* currentDate = localtime(&currentTime);

		char out_name[1024];
#ifdef LIBPNG_SUPPORT
		sprintf(out_name, "histogram-%d-%02d-%02d_%02d-%02d-%02d.png", currentDate->tm_year+1900, currentDate->tm_mon+1, currentDate->tm_mday, currentDate->tm_hour, currentDate->tm_min, currentDate->tm_sec);
#else
#ifdef LIBJPEG_SUPPORT
		sprintf(out_name, "histogram-%d-%02d-%02d_%02d-%02d-%02d.jpg", currentDate->tm_year + 1900, currentDate->tm_mon + 1, currentDate->tm_mday, currentDate->tm_hour, currentDate->tm_min, currentDate->tm_sec);
#else
		sprintf(out_name, "histogram-%d-%02d-%02d_%02d-%02d-%02d.pnm", currentDate->tm_year + 1900, currentDate->tm_mon + 1, currentDate->tm_mday, currentDate->tm_hour, currentDate->tm_min, currentDate->tm_sec);
#endif
#endif

		out.WriteImage(out_name);
	}

	// fix scale
	float max_extent = std::max(volumeSize.width * scale.x, std::max(volumeSize.height * scale.y, volumeSize.depth * scale.z));

	scale.x = volumeSize.width  * scale.x / max_extent;
	scale.y = volumeSize.height * scale.y / max_extent;
	scale.z = volumeSize.depth  * scale.z / max_extent;

	// stacked volume
	volumeSize.depth *= element_count;
	scale.z *= element_count;
	g_components = element_count;

	if (ref_file)
	{
		// use command-line specified CUDA device, otherwise use device with highest Gflops/s
		chooseCudaDevice(argc, (const char **)argv, false);
	}
	else
	{
		// First initialize OpenGL context, so we can properly set the GL for CUDA.
		// This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
		initGL(&argc, argv);

		// use command-line specified CUDA device, otherwise use device with highest Gflops/s
		chooseCudaDevice(argc, (const char **)argv, true);
	}

	size_t cuda_free, cuda_total;
	cudaMemGetInfo(&cuda_free, &cuda_total);
	std::cout << "Total  CUDA memory: " << cuda_total << std::endl;
	std::cout << "Free   CUDA memory: " << cuda_free << std::endl;

	printf("Uploading volume data\n");

	if (originalSize.width == 0) originalSize.width = volumeSize.width;
	if (originalSize.height == 0) originalSize.height = volumeSize.height;
	if (originalSize.depth == 0) originalSize.depth = volumeSize.depth;

	if (volumeType == 0)
	{
		renderer8->initCuda((VolumeType8 *)raw_volume, volumeSize, originalSize, scale, element_count, max_error);
		renderer8->setTextureFilterMode(linearFiltering);
		renderer8->enableLighting(lighting);
		renderer8->enableRGBA(rgbColor);
	}
	else if (volumeType == 1)
	{
		renderer16->initCuda((VolumeType16 *)raw_volume, volumeSize, originalSize, scale, element_count, max_error);
		renderer16->setTextureFilterMode(linearFiltering);
		renderer16->enableLighting(lighting);
		renderer16->enableRGBA(rgbColor);
	}
	else if (volumeType == 2)
	{
		renderer32->initCuda((VolumeType32 *)raw_volume, volumeSize, originalSize, scale, element_count, max_error);
		renderer32->setTextureFilterMode(linearFiltering);
		renderer32->enableLighting(lighting);
		renderer32->enableRGBA(rgbColor);
	}
	else
	{
		renderer64->initCuda((VolumeType64 *)raw_volume, volumeSize, originalSize, scale, element_count, max_error);
		renderer64->setTextureFilterMode(linearFiltering);
		renderer64->enableLighting(lighting);
		renderer64->enableRGBA(rgbColor);
	}
	free(raw_volume);

	for (size_t i = 0; i < hist.size(); i++)
	{
		std::cout << i << ": " << hist[i] << std::endl;
	}

	g_scale = scale;

	sdkCreateTimer(&timer);

	printf("Press '+' and '-' to change density (0.01 increments)\n"
		   "      ']' and '[' to change brightness\n"
		   "      ';' and ''' to modify transfer function offset\n"
		   "      '.' and ',' to modify transfer function scale\n"
		   "      'f'         toggle filter\n"
		   "      'b'         toggle background\n"
		   "      'c'         toggle rgba color volume\n"
		   "      'l'         toggle lighting (includes gradient magnitude modulation with color volume on)\n\n");

	// calculate new grid size
	gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));

	if (ref_file)
	{
		runSingleTest(ref_file, argv[0]);
	}
	else
	{
		// This is the normal rendering path for VolumeRender
		glutDisplayFunc(display);
		glutKeyboardFunc(keyboard);
		glutMouseFunc(mouse);
		glutMotionFunc(motion);
		glutReshapeFunc(reshape);
		glutIdleFunc(idle);

		initPixelBuffer();

		atexit(cleanup);

		glutMainLoop();
	}

	checkCudaErrors(cudaDeviceReset());
	exit(EXIT_SUCCESS);
}
