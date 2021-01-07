// Copyright (c) 2016 Stefan Guthe / GCC / TU-Darmstadt. All rights reserved. 
// Use of this source code is governed by the BSD 3-Clause license that can be
// found in the LICENSE file.

#pragma once
#include "cuda_runtime.h"

typedef unsigned char VolumeType8;
typedef unsigned short VolumeType16;
typedef uchar4 VolumeType32;
typedef ushort4 VolumeType64;

template <class T>
T *loadRawFile(char *filename, size_t size, float3 &scale, int raw_skip);

#ifdef LIBPNG_SUPPORT
template <class T>
T* loadPngFiles(char *filename, cudaExtent &volumeSize, float3 &scale, int start, int end, int clip_x0, int clip_x1, int clip_y0, int clip_y1, float scale_png, bool clip_zero);

unsigned int getPngElementSize(char *filename, int start);
unsigned int getPngComponents(char *filename, int start);
#endif

void *loadDatFile(char *filename, cudaExtent &volumeSize, float3 &scale, unsigned int &elementSize, unsigned int &components);

void saveDatFile(char *export_name, cudaExtent &volumeSize, float3 &scale, unsigned int &element_size, unsigned int &element_count, void *raw_volume, int volumeType, int export_version);

template <class T>
void denoiseVolume(T *vol, cudaExtent &volumeSize, int denoise);

template <class T, class A, class D, class O>
void quantizeVolume(T *vol, cudaExtent &volumeSize, int lossy, bool bruteForce);

template <class T>
void resampleVolume(T *vol, T *out, cudaExtent &volumeSize, cudaExtent &resampleSize);

template <class T>
void expandVolume(T *vol, T *out, cudaExtent &volumeSize, cudaExtent &resampleSize);

template <class T>
T *calculateGradients(T *vol, cudaExtent &volumeSize, unsigned int &components);

template <typename T>
void printSize(T *vol, cudaExtent volumeSize, unsigned int components);
