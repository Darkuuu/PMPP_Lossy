// Copyright (c) 2016 Stefan Guthe / GCC / TU-Darmstadt. All rights reserved. 
// Use of this source code is governed by the BSD 3-Clause license that can be
// found in the LICENSE file.

#define _CRT_SECURE_NO_WARNINGS

#include "pnm_image.h"
#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <helper_string.h>

int PnmImage::ReadValue(char *buffer, FILE *fp)
{
	fread(buffer, 1, 1, fp);
	while ((buffer[0] == ' ') || (buffer[0] == 0x0a) || (buffer[0] == 0x0d)) fread(buffer, 1, 1, fp);
	int len = 1;
	fread(&(buffer[len]), 1, 1, fp);
	while ((buffer[len] != ' ') || (buffer[len] != 0x0a) || (buffer[len] != 0x0d))
	{
		len++;
		fread(&(buffer[len]), 1, 1, fp);
	}
	buffer[len] = 0;
	return len;
}

int PnmImage::GetInteger(char *buffer, int len)
{
	int value = 0;
	for (int i = 0; i < len; i++)
	{
		value *= 10;
		value += (int)buffer[i] - (int)'0';
	}
	return value;
}

bool PnmImage::ReadImage(const char fileName[])
{
#pragma omp critical
	{
		std::cout << "Reading image \"" << fileName << "\"" << std::endl;
	}

	Cleanup();

	FILE *fp = fopen(fileName, "rb");

	if (fp == NULL) return false;

	char buffer[2048];
	fread(buffer, 1, 2, fp);

	if (buffer[0] != 'P')
	{
		fclose(fp);
		return false;
	}

	bool ascii;
	bool extended = false;
	switch (buffer[1])
	{
	case 1:
		ascii = true;
		m_bitDepth = 1;
		m_components = 1;
		break;
	case 2:
		ascii = true;
		m_bitDepth = 8;
		m_components = 1;
		break;
	case 3:
		ascii = true;
		m_bitDepth = 8;
		m_components = 3;
		break;
	case 4:
		ascii = false;
		m_bitDepth = 1;
		m_components = 1;
		break;
	case 5:
		ascii = false;
		m_bitDepth = 8;
		m_components = 1;
		break;
	case 6:
		ascii = false;
		m_bitDepth = 8;
		m_components = 3;
		break;
	case 7:
		ascii = false;
		extended = true;
		break;
	default:
		fclose(fp);
		return false;
	}

	if (!extended)
	{
		int len = ReadValue(buffer, fp);
		m_width = GetInteger(buffer, len);
		len = ReadValue(buffer, fp);
		m_height = GetInteger(buffer, len);
		if (m_bitDepth != 1)
		{
			len = ReadValue(buffer, fp);
			int maximum = GetInteger(buffer, len);
			m_mask = 1;
			while ((int)m_mask < maximum) m_mask = (m_mask << 1) + 1;
		}
		else
		{
			m_mask = 1;
		}
	}
	else
	{
		bool done_reading = false;
		while (!done_reading)
		{
			int len = ReadValue(buffer, fp);
			if (STRCASECMP(buffer, "width") == 0)
			{
				len = ReadValue(buffer, fp);
				m_width = GetInteger(buffer, len);
			}
			else if (STRCASECMP(buffer, "height") == 0)
			{
				len = ReadValue(buffer, fp);
				m_height = GetInteger(buffer, len);
			}
			else if (STRCASECMP(buffer, "depth") == 0)
			{
				len = ReadValue(buffer, fp);
				m_components = GetInteger(buffer, len);
			}
			else if (STRCASECMP(buffer, "maxval") == 0)
			{
				len = ReadValue(buffer, fp);
				int maximum = GetInteger(buffer, len);
				m_mask = 1;
				while ((int)m_mask < maximum) m_mask = (m_mask << 1) + 1;
			}
			else if (STRCASECMP(buffer, "tupltype") == 0)
			{
				len = ReadValue(buffer, fp);
				// don't care
			}
			else if (STRCASECMP(buffer, "endhdr") == 0)
			{
				done_reading = false;
			}
			else
			{
				fclose(fp);
				return false;
			}
		}
	}

	if (!Alloc())
	{
		fclose(fp);
		return false;
	}

	if (ascii)
	{
		for (unsigned int y = 0; y < m_height; y++)
		{
			for (unsigned int x = 0; x < m_width; x++)
			{
				for (unsigned int c = 0; c < m_components; c++)
				{
					int len = ReadValue(buffer, fp);
					m_data[(x + y * m_width) * m_components + c] = GetInteger(buffer, len);
				}
			}
		}
	}
	else if ((m_bitDepth == 1) && (!extended))
	{
		for (unsigned int y = 0; y < m_height; y++)
		{
			int avail = 0;
			for (unsigned int x = 0; x < m_width; x++)
			{
				for (unsigned int c = 0; c < m_components; c++)
				{
					if (avail == 0)
					{
						fread(buffer, 1, 1, fp);
						avail = 8;
					}
					// msb first, packed bits
					m_data[(x + y * m_width) * m_components + c] = ((buffer[0] & 0x80) == 0) ? 1 : 0;
					buffer[0] <<= 1;
					avail--;
				}
			}
		}
	}
	else if (m_bitDepth <= 8)
	{
		for (unsigned int y = 0; y < m_height; y++)
		{
			for (unsigned int x = 0; x < m_width; x++)
			{
				for (unsigned int c = 0; c < m_components; c++)
				{
					fread(&(m_data[(x + y * m_width) * m_components + c]), 1, 1, fp);
				}
			}
		}
	}
	else
	{
		for (unsigned int y = 0; y < m_height; y++)
		{
			for (unsigned int x = 0; x < m_width; x++)
			{
				for (unsigned int c = 0; c < m_components; c++)
				{
					fread(buffer, 2, 1, fp);
					// msb first!
					std::swap(buffer[0], buffer[1]);
					((unsigned short *)m_data)[(x + y * m_width) * m_components + c] = ((unsigned short *)buffer)[0];
				}
			}
		}
	}

	fclose(fp);
	return true;
}

bool PnmImage::WriteImage(const char fileName[])
{
	std::cout << "Writing image \"" << fileName << "\"" << std::endl;

	int len = (int)strlen(fileName);

	bool extended;
	if (((fileName[len - 3] == 'P') || (fileName[len - 3] == 'p')) &&
		((fileName[len - 2] == 'A') || (fileName[len - 2] == 'a')) &&
		((fileName[len - 1] == 'M') || (fileName[len - 1] == 'm'))) extended = true; else extended = false;

	FILE *fp = fopen(fileName, "w");
	if (fp == NULL) return false;

	if (!extended)
	{
		// write binary formats only
		int mode;
		if ((m_bitDepth == 1) && (m_components == 1)) mode = 4;
		else if (m_components == 1) mode = 5;
		else mode = 6;
		fprintf(fp, "P%d\n", mode);
		fprintf(fp, "%d %d\n", m_width, m_height);
		if (mode != 4) fprintf(fp, "%d", m_mask);
	}
	else
	{
		fprintf(fp, "P7\n");
		fprintf(fp, "WIDTH %d\n", m_width);
		fprintf(fp, "HEIGHT %d\n", m_height);
		fprintf(fp, "DEPTH %d\n", m_components);
		fprintf(fp, "MAXVAL %d", m_mask);
		if ((m_bitDepth == 1) && (m_components == 1)) fprintf(fp, "TUPLTYPE BLACKANDWHITE\n");
		else if ((m_bitDepth == 1) && (m_components == 2)) fprintf(fp, "TUPLTYPE BLACKANDWHITE_ALPHA\n");
		else if (m_components == 1) fprintf(fp, "TUPLTYPE GRAYSCALE\n");
		else if (m_components == 2) fprintf(fp, "TUPLTYPE GRAYSCALE_ALPHA\n");
		else if (m_components == 3) fprintf(fp, "TUPLTYPE RGB\n");
		else fprintf(fp, "TUPLTYPE RGB_ALPHA\n");
		fprintf(fp, "ENDHDR\n");
	}
	fclose(fp);

	fp = fopen(fileName, "ab");
	char buffer[2];
	if ((m_bitDepth == 1) && (!extended))
	{
		for (unsigned int y = 0; y < m_height; y++)
		{
			int avail = 0;
			buffer[0] = 0;
			for (unsigned int x = 0; x < m_width; x++)
			{
				for (unsigned int c = 0; c < m_components; c++)
				{
					if (avail == 8)
					{
						fwrite(buffer, 1, 1, fp);
						avail = 0;
						buffer[0] = 0;
					}
					// msb first, packed bits
					char bit = (m_data[(x + y * m_width) * m_components + c]==0) ? 1 : 0;
					buffer[0] |= 1 << (7 - avail);
					avail++;
				}
			}
			if (avail > 0) fwrite(buffer, 1, 1, fp);
		}
	}
	else if (m_bitDepth <= 8)
	{
		for (unsigned int y = 0; y < m_height; y++)
		{
			for (unsigned int x = 0; x < m_width; x++)
			{
				for (unsigned int c = 0; c < m_components; c++)
				{
					fwrite(&(m_data[(x + y * m_width) * m_components + c]), 1, 1, fp);
				}
			}
		}
	}
	else
	{
		for (unsigned int y = 0; y < m_height; y++)
		{
			for (unsigned int x = 0; x < m_width; x++)
			{
				for (unsigned int c = 0; c < m_components; c++)
				{
					((unsigned short *)buffer)[0] = ((unsigned short *)m_data)[(x + y * m_width) * m_components + c];
					// msb first!
					std::swap(buffer[0], buffer[1]);
					fwrite(buffer, 2, 1, fp);
				}
			}
		}
	}
	fclose(fp);

	return true;
}

void PnmImage::Cleanup()
{
	if (m_data != NULL) {
		free(m_data);
	}
	m_data = NULL;
}

bool PnmImage::Alloc()
{
	if ((m_width == 0) || (m_height == 0) || (m_components == 0)) return false;
	unsigned int num_bytes = m_components * ((m_bitDepth == 16)?2:1);
	m_data = (unsigned char*)malloc(num_bytes * m_width * m_height);
	return m_data != NULL;
}

void PnmImage::SetBitDepth(unsigned int bitDepth) {
	// pnm only allows for a m_bitDepth of 1, 8 or 16
	if (bitDepth == 1) m_bitDepth = 1;
	else if (bitDepth <= 8) m_bitDepth = 8;
	else m_bitDepth = 16;

	m_mask = ~((~0u) << bitDepth);
	Alloc();
}

void PnmImage::VSetValue(unsigned int x, unsigned int y, unsigned int c, unsigned int v)
{
	if (m_data == 0) return;
	if (x >= m_width) return;
	if (y >= m_height) return;
	if (c >= m_components) return;
	if (m_bitDepth < 16)
	{
		m_data[(x + y * m_width) * m_components + c] = std::min(255U >> (8 - m_bitDepth), v);
	}
	else
	{
		((unsigned short *)m_data)[(x + y * m_width) * m_components + c] = std::min(65535U, v);
	}
}

const unsigned int PnmImage::VGetValue(unsigned int x, unsigned int y, unsigned int c) const
{
	if (m_data == 0) return 0;
	if (x >= m_width) x = m_width - 1;
	if (y >= m_height) y = m_height - 1;
	if (c >= m_components) return 0;
	if (m_bitDepth < 16)
	{
		return m_data[(x + y * m_width) * m_components + c];
	}
	else
	{
		return ((unsigned short *)m_data)[(x + y * m_width) * m_components + c];
	}
}
