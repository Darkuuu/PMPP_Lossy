// Copyright (c) 2016 Stefan Guthe / GCC / TU-Darmstadt. All rights reserved. 
// Use of this source code is governed by the BSD 3-Clause license that can be
// found in the LICENSE file.

#include "png_image.h"

#ifdef LIBPNG_SUPPORT

#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <stdlib.h>
#include <algorithm>

bool PngImage::ReadImage(const char fileName[])
{
#pragma omp critical
	{
		std::cout << "Reading image \"" << fileName << "\"" << std::endl;
	}

	Cleanup();

	png_byte color_type;

	png_structp png_ptr;
	png_infop info_ptr;
	int number_of_passes;

	png_byte header[8];    // 8 is the maximum size that can be checked

	/* open file and test for it being a png */
	FILE *fp = fopen(fileName, "rb");
	if (!fp) {
		std::cerr << "[read_png_file] File " << fileName << " could not be opened for reading" << std::endl;
		return false;
	}

	size_t tmp = fread(header, 1, 8, fp);
	if ((png_sig_cmp(header, 0, 8)) || (tmp == (size_t) 0)) {
		std::cerr << "[read_png_file] File " << fileName << " is not recognized as a PNG file" << std::endl;
		fclose(fp);
		return false;
	}


	/* initialize stuff */
	png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

	if (!png_ptr) {
		std::cerr << "[read_png_file] png_create_read_struct failed" << std::endl;
		fclose(fp);
		return false;
	}

	info_ptr = png_create_info_struct(png_ptr);
	if (!info_ptr) {
		std::cerr << "[read_png_file] png_create_info_struct failed" << std::endl;
		free(png_ptr);
		fclose(fp);
		return false;
	}

	if (setjmp(png_jmpbuf(png_ptr))) {
		std::cerr << "[read_png_file] Error during init_io" << std::endl;
		free(info_ptr);
		free(png_ptr);
		fclose(fp);
		return false;
	}

	png_init_io(png_ptr, fp);
	png_set_sig_bytes(png_ptr, 8);

	png_read_info(png_ptr, info_ptr);

	m_width     = png_get_image_width (png_ptr, info_ptr);
	m_height    = png_get_image_height(png_ptr, info_ptr);
	color_type      = png_get_color_type  (png_ptr, info_ptr);
	m_bitDepth = png_get_bit_depth   (png_ptr, info_ptr);
	m_mask     = ~((~0u) << m_bitDepth);

	number_of_passes = png_set_interlace_handling(png_ptr);
	png_read_update_info(png_ptr, info_ptr);


	/* read file */
	if (setjmp(png_jmpbuf(png_ptr))) {
		std::cerr << "[read_png_file] Error during read_image" << std::endl;
		free(info_ptr);
		free(png_ptr);
		fclose(fp);
		return false;
	}

	m_rowPointer = (png_bytep*) malloc(sizeof(png_bytep) * m_height);
	for (unsigned int y=0; y < m_height; y++) {
		m_rowPointer[y] = (png_byte*) malloc(png_get_rowbytes(png_ptr, info_ptr));
	}

	png_read_image(png_ptr, m_rowPointer);

	fclose(fp);

	m_components = 0;
	switch (color_type) {
		case PNG_COLOR_TYPE_GRAY:
			m_components = 1;
			break;
		case PNG_COLOR_TYPE_GRAY_ALPHA:
			m_components = 2;
			break;
		case PNG_COLOR_TYPE_RGB:
			m_components = 3;
			break;
		case PNG_COLOR_TYPE_RGBA:
			m_components = 4;
			break;
		default:
			std::cerr << "[process_file] color_type of input file must be PNG_COLOR_TYPE_RGBA (" << PNG_COLOR_TYPE_RGBA <<
						") or PNG_COLOR_TYPE_RGB (" << PNG_COLOR_TYPE_RGB << ") (is " << color_type << ")" << std::endl;
	}

	free(png_ptr);
	free(info_ptr);

	return (m_components != 0);
}

bool PngImage::WriteImage(const char fileName[])
{
	std::cout << "Writing image \"" << fileName << "\"" << std::endl;

	png_byte color_type;

	png_structp png_ptr;
	png_infop info_ptr;

	switch (m_components) {
		case 1:
			color_type = PNG_COLOR_TYPE_GRAY;
			break;
		case 2:
			color_type = PNG_COLOR_TYPE_GRAY_ALPHA;
			break;
		case 3:
			color_type = PNG_COLOR_TYPE_RGB;
			break;
		case 4:
			color_type = PNG_COLOR_TYPE_RGBA;
			break;
		default:
			std::cerr << "[process_file] number of components must be 1-4 in order to produce a meaningful color_type" << std::endl;
	}


	/* create file */
	FILE *fp = fopen(fileName, "wb");
	if (!fp) {
		std::cerr << "[write_png_file] File " << fileName << " could not be opened for writing" << std::endl;
		return false;
	}

	/* initialize stuff */
	png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

	if (!png_ptr) {
		std::cerr << "[write_png_file] png_create_write_struct failed" << std::endl;
		fclose(fp);
		return false;
	}

	info_ptr = png_create_info_struct(png_ptr);
	if (!info_ptr) {
		std::cerr << "[write_png_file] png_create_info_struct failed" << std::endl;
		free(png_ptr);
		fclose(fp);
		return false;
	}

	if (setjmp(png_jmpbuf(png_ptr))) {
		std::cerr << "[write_png_file] Error during init_io" << std::endl;
		free(info_ptr);
		free(png_ptr);
		fclose(fp);
		return false;
	}

	png_init_io(png_ptr, fp);


	/* write header */
	if (setjmp(png_jmpbuf(png_ptr))) {
		std::cerr << "[write_png_file] Error during writing header" << std::endl;
		free(info_ptr);
		free(png_ptr);
		fclose(fp);
		return false;
	}

	png_set_IHDR(png_ptr, info_ptr, m_width, m_height,
		     m_bitDepth, color_type, PNG_INTERLACE_NONE,
		     PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

	png_write_info(png_ptr, info_ptr);


	/* write bytes */
	if (setjmp(png_jmpbuf(png_ptr))) {
		std::cerr << "[write_png_file] Error during writing bytes" << std::endl;
		free(info_ptr);
		free(png_ptr);
		fclose(fp);
		return false;
	}

	png_write_image(png_ptr, m_rowPointer);


	/* end write */
	if (setjmp(png_jmpbuf(png_ptr))) {
		std::cerr << "[write_png_file] Error during end of write" << std::endl;
		fclose(fp);
		free(png_ptr);
		free(info_ptr);
		return false;
	}


	png_write_end(png_ptr, NULL);

	fclose(fp);
	free(png_ptr);
	free(info_ptr);
	return true;
}

void PngImage::Cleanup()
{
	if (m_rowPointer != NULL) {
		for (unsigned int y=0; y<m_height; y++) {
			if (m_rowPointer[y] != NULL) {
				free(m_rowPointer[y]);
			}
		}
		free(m_rowPointer);
		m_rowPointer = NULL;
	}
}

bool PngImage::Alloc()
{
	if ((m_width == 0) || (m_height == 0) || (m_components == 0)) return false;
	unsigned int num_bytes = m_components * ((m_bitDepth == 16)?2:1);
	m_rowPointer = (png_bytep*) malloc(sizeof(png_bytep) * m_height);
	if (m_rowPointer == NULL) return false;
	memset(m_rowPointer, 0, sizeof(sizeof(png_bytep) * m_height));
	for (unsigned int y=0; y < m_height; y++) {
		m_rowPointer[y] = (png_byte*) malloc(m_width * num_bytes);
		if (m_rowPointer[y] == NULL) {
			Cleanup();
			return false;
		}
	}
	return true;
}

void PngImage::SetBitDepth(unsigned int m_bitDepth) {
	// png only allows for a m_bitDepth of 1, 2, 4, 8 or 16
	this->m_bitDepth = 1;
	while ((this->m_bitDepth < 16) && (this->m_bitDepth < m_bitDepth)) this->m_bitDepth <<= 1;
	m_mask = ~((~0u) << m_bitDepth);
	Alloc();
}

void PngImage::VSetValue(unsigned int x, unsigned int y, unsigned int c, unsigned int v)
{
	if (m_rowPointer == 0) return;
	if (x >= m_width) return;
	if (y >= m_height) return;
	if (c >= m_components) return;
	if (m_bitDepth < 16)
	{
		m_rowPointer[y][x * m_components + c] = std::min(255U >> (8 - m_bitDepth), v);
	}
	else
	{
		m_rowPointer[y][(x * m_components + c) * 2]     = std::min(65535U, v) >> 8;
		m_rowPointer[y][(x * m_components + c) * 2 + 1] = std::min(65535U, v) & 255U;
	}
}

const unsigned int PngImage::VGetValue(unsigned int x, unsigned int y, unsigned int c) const
{
	if (m_rowPointer == 0) return 0;
	if (x >= m_width) x = m_width - 1;
	if (y >= m_height) y = m_height - 1;
	if (c >= m_components) return 0;
	if (m_bitDepth < 16)
	{
		return m_rowPointer[y][x * m_components + c];
	}
	else
	{
		return (((unsigned int) m_rowPointer[y][(x * m_components + c) * 2]) << 8) + m_rowPointer[y][(x * m_components + c) * 2 + 1];
	}
}

#endif
