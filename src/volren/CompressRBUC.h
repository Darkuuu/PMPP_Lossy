// Copyright (c) 2016 Stefan Guthe / GCC / TU-Darmstadt. All rights reserved. 
// Use of this source code is governed by the BSD 3-Clause license that can be
// found in the LICENSE file.

#pragma once

#include "global_defines.h"

template <typename T>
unsigned int compressRBUC8x8(T *in, unsigned char *comp);

template <typename T>
unsigned int decompressRBUC8x8(unsigned char *comp, T *out);

