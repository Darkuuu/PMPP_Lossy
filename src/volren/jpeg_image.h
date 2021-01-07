// Copyright (c) 2016 Stefan Guthe / GCC / TU-Darmstadt. All rights reserved. 
// Use of this source code is governed by the BSD 3-Clause license that can be
// found in the LICENSE file.

#pragma once

#include "global_defines.h"
#include "../../config.h"

#ifdef LIBJPEG_SUPPORT

#include "image.h"
#include <jpeglib.h>

// compressed integer image format (usually 8 bit per channel; specified in depth)
class JpegImage : public Image {
private:
	JSAMPROW *m_rowPointer;
	int m_quality;

private:
	void Cleanup();

	bool Alloc();

public:
	JpegImage() : m_rowPointer(NULL), m_quality(95) { m_mask = 255; }

	virtual ~JpegImage() { Cleanup(); }

public:
	virtual bool ReadImage(const char fileName[]) override;
	
	void SetQuality(int quality) { m_quality = quality; }
	
	virtual bool WriteImage(const char fileName[]) override;

	virtual void SetWidth(unsigned int width) override { m_width = width; Alloc(); }

	virtual void SetHeight(unsigned int height) override { m_height = height; Alloc(); }

	virtual void SetComponents(unsigned int components) override { m_components = components; Alloc(); }

	virtual void VSetValue(unsigned int x, unsigned int y, unsigned int c, unsigned int v) override;

	virtual const unsigned int VGetValue(unsigned int x, unsigned int y, unsigned int c) const override;
};

#endif
