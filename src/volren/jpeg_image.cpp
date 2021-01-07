// Copyright (c) 2016 Stefan Guthe / GCC / TU-Darmstadt. All rights reserved. 
// Use of this source code is governed by the BSD 3-Clause license that can be
// found in the LICENSE file.

#include "jpeg_image.h"

#ifdef LIBJPEG_SUPPORT

#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <stdlib.h>
#include <algorithm>

bool JpegImage::ReadImage(const char fileName[])
{
	std::cout << "Reading image \"" << fileName << "\"" << std::endl;

	Cleanup();

		/* these are standard libjpeg structures for reading(decompression) */
	struct jpeg_decompress_struct cinfo;
	struct jpeg_error_mgr jerr;
	
	FILE *infile = fopen( fileName, "rb" );
	unsigned long location = 0;
	int i = 0;
	
	if ( !infile )
	{
		std::cerr << "Error opening jpeg file " << fileName << std::endl;
		return false;
	}
	/* here we set up the standard libjpeg error handler */
	cinfo.err = jpeg_std_error( &jerr );
	/* setup decompression process and source, then read JPEG header */
	jpeg_create_decompress( &cinfo );
	/* we want some "extra" stuff */
	cinfo.dct_method = JDCT_FLOAT;
	/* this makes the library read from infile */
	jpeg_stdio_src( &cinfo, infile );
	/* reading the image header which contains image information */
	jpeg_read_header( &cinfo, TRUE );
	/* Uncomment the following to output image information, if needed. */
	/*--
	printf( "JPEG File Information: \n" );
	printf( "Image width and height: %d pixels and %d pixels.\n", cinfo.image_width, cinfo.image_height );
	printf( "Color components per pixel: %d.\n", cinfo.num_components );
	printf( "Color space: %d.\n", cinfo.jpeg_color_space );
	--*/
	/* Start decompression jpeg here */
	jpeg_start_decompress( &cinfo );
	m_components = cinfo.num_components;
	m_width = cinfo.output_width;
	m_height = cinfo.output_height;

	if (!Alloc()) {
		std::cerr << "jpeg m_rowPointer allocation failed!" << std::endl;
		jpeg_finish_decompress( &cinfo );
		jpeg_destroy_decompress( &cinfo );
		fclose( infile );
		return false;
	}

	/* read scan lines */
	unsigned int read = 0;
	while (read < m_height)
	{
		read += jpeg_read_scanlines( &cinfo, &(m_rowPointer[read]), m_height );
	}
	/* wrap up decompression, destroy objects, free pointers and close open files */
	jpeg_finish_decompress( &cinfo );
	jpeg_destroy_decompress( &cinfo );
	fclose( infile );

	/* yup, we succeeded! */
	return true;
}

bool JpegImage::WriteImage(const char fileName[])
{
	std::cout << "Writing image \"" << fileName << "\"" << std::endl;

	J_COLOR_SPACE color_space;
	if (m_components == 1)
	{
		color_space = JCS_GRAYSCALE;
	}
	else
	{
		color_space = JCS_RGB;
	}

	struct jpeg_compress_struct cinfo;
	struct jpeg_error_mgr jerr;
	
	/* this is a pointer to one row of image data */
	FILE *outfile = fopen( fileName, "wb" );
	
	if ( !outfile )
	{
		std::cerr << "Error opening output jpeg file " << fileName << std::endl;
		return false;
	}
	cinfo.err = jpeg_std_error( &jerr );
	jpeg_create_compress(&cinfo);
	jpeg_stdio_dest(&cinfo, outfile);

	/* Setting the parameters of the output file here */
	cinfo.image_width = m_width;	
	cinfo.image_height = m_height;
	cinfo.input_components = m_components;
	cinfo.in_color_space = color_space;
    /* default compression parameters, we shouldn't be worried about these */
	jpeg_set_defaults( &cinfo );
	/* we want some "extra" stuff */
	jpeg_set_quality(&cinfo, m_quality, TRUE);
	cinfo.optimize_coding = TRUE;
	cinfo.dct_method = JDCT_FLOAT;
	/* Now do the compression .. */
	jpeg_start_compress( &cinfo, TRUE );
	/* like reading a file */
	jpeg_write_scanlines( &cinfo, m_rowPointer, m_height );
	/* similar to read file, clean up after we're done compressing */
	jpeg_finish_compress( &cinfo );
	jpeg_destroy_compress( &cinfo );
	fclose( outfile );
	/* success */
	return true;
}

void JpegImage::Cleanup()
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

bool JpegImage::Alloc()
{
	Cleanup();
	if ((m_width == 0) || (m_height == 0) || (m_components == 0)) return false;
	unsigned int num_bytes = ((m_components == 1)?1:3) * sizeof(JSAMPLE);
	m_rowPointer = (JSAMPROW *) malloc(sizeof(JSAMPROW *) * m_height);
	if (m_rowPointer == NULL) return false;
	memset(m_rowPointer, 0, sizeof(sizeof(JSAMPROW *) * m_height));
	for (unsigned int y=0; y < m_height; y++) {
		m_rowPointer[y] = (JSAMPROW) malloc(m_width * num_bytes);
		if (m_rowPointer[y] == NULL) {
			Cleanup();
			return false;
		}
	}
	return true;
}

void JpegImage::VSetValue(unsigned int x, unsigned int y, unsigned int c, unsigned int v)
{
	if (m_rowPointer == 0) return;
	if (x >= m_width) return;
	if (y >= m_height) return;
	if (c >= m_components) return;
	m_rowPointer[y][x * m_components + c] = std::min(255U, v);
}

const unsigned int JpegImage::VGetValue(unsigned int x, unsigned int y, unsigned int c) const
{
	if (m_rowPointer == 0) return 0;
	if (x >= m_width) x = m_width - 1;
	if (y >= m_height) y = m_height - 1;
	if (c >= m_components) return 0;
	return m_rowPointer[y][x * m_components + c];
}

#endif