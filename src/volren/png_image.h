// Copyright (c) 2016 Stefan Guthe / GCC / TU-Darmstadt. All rights reserved. 
// Use of this source code is governed by the BSD 3-Clause license that can be
// found in the LICENSE file.

#pragma once

#include "global_defines.h"
#include "../../config.h"

#ifdef LIBPNG_SUPPORT

#include "image.h"

// compressed integer image format (usually 8 bit per channel; specified in depth)
class PngImage : public Image {
private:
	unsigned int m_bitDepth;
	png_bytep *m_rowPointer;

private:
	void Cleanup();

	bool Alloc();

public:
	PngImage() : m_rowPointer(NULL), m_bitDepth(8) { m_mask = 255; }

	virtual ~PngImage() { Cleanup(); }

public:
	virtual bool ReadImage(const char fileName[]) override;
	
	void SetBitDepth(unsigned int bit_depth);

	unsigned int GetBitDepth() { return m_bitDepth; }

	virtual bool WriteImage(const char fileName[]) override;

	virtual void SetWidth(unsigned int width) override { Cleanup(); m_width = width; Alloc(); }

	virtual void SetHeight(unsigned int height) override { Cleanup(); m_height = height; Alloc(); }

	virtual void SetComponents(unsigned int components) override { Cleanup(); m_components = components; Alloc(); }

	virtual void VSetValue(unsigned int x, unsigned int y, unsigned int c, unsigned int v) override;

	virtual const unsigned int VGetValue(unsigned int x, unsigned int y, unsigned int c) const override;
};

#endif
