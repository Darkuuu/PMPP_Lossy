// Copyright (c) 2016 Stefan Guthe / GCC / TU-Darmstadt. All rights reserved. 
// Use of this source code is governed by the BSD 3-Clause license that can be
// found in the LICENSE file.

/*
 * If you find the source code provided useful in any way, please send us a short
 * email to stefan.guthe@gris.informatik.tu-darmstadt.de
 * If not, please send us comments as to how to improve this release.
 */

#pragma once

// these require external libraries that are not bundled with this release

// requires codebase.h, ddsbase.h and ddsbase.cpp from Stefan Roettgers volume renderer V^3 http://www.stereofx.org/
// put files into src/v3 for automatic building
//#define PVM_SUPPORT

// requires zlib
#define ZLIB_SUPPORT

// requires libjepg
#define LIBJPEG_SUPPORT

// requires libpng
#ifdef ZLIB_SUPPORT
#define LIBPNG_SUPPORT
#endif
