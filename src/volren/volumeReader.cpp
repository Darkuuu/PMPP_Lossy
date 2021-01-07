// Copyright (c) 2016 Stefan Guthe / GCC / TU-Darmstadt. All rights reserved. 
// Use of this source code is governed by the BSD 3-Clause license that can be
// found in the LICENSE file.

#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX

#include <algorithm>
#include "volumeReader.h"
#include "png_image.h"
#include <vector>
#include <queue>
#include <chrono>
#include "global_defines.h"
#include "../../config.h"
#include "CompressRBUC.h"

typedef unsigned int  uint;
typedef unsigned char uchar;

#ifdef WIN32
#include <Windows.h>
#endif
#include <iostream>

#ifdef PVM_SUPPORT
#include "../v3/ddsbase.cpp"
#endif

#ifdef ZLIB_SUPPORT
#include "zlib.h"
#endif

#ifdef LIBPNG_SUPPORT
#include "png_image.h"
#endif

// included for writing ktx format
#include <GL/glew.h>

// arithmetic compression/decompression
#include "arithcoder.h"

// file class to combine compressed (if enabled) and regular file access
class VolumeFile
{
private:
	bool compressed;
	FILE *fp;
#ifdef ZLIB_SUPPORT
	gzFile f;
#endif
	bool RLE;
	bool arithmetic;
	std::deque<unsigned char> RLE_buffer;
	Encoder<uint64> encoder;
	Decoder<uint64> decoder;
	Context<uint64, unsigned short> C;
	bool writing;

	int read_internal(void *buffer, unsigned int len)
	{
#ifdef ZLIB_SUPPORT
		if (compressed)
		{
			return gzread(f, buffer, len);
		}
		else
#endif
		{
			return (int)fread(buffer, sizeof(char), len, fp);
		}
	}

	int write_internal(void *buffer, unsigned int len)
	{
#ifdef ZLIB_SUPPORT
		if (compressed)
		{
			return gzwrite(f, buffer, len);
		}
		else
#endif
		{
			return (int)fwrite(buffer, sizeof(char), len, fp);
		}
	}

	void read_RLE()
	{
		// 0 is followed by repeat count
		unsigned char sym;
		read_internal(&sym, 1);
		if (sym == 0)
		{
			read_internal(&sym, 1);
			int count = (int)((unsigned char)sym);
			count += 1;
			for (int c = 0; c < count; c++) RLE_buffer.push_back(0);
		}
		else
		{
			RLE_buffer.push_back(sym);
		}
	}

	void write_RLE(bool flush)
	{
		// buffer at least 512 bytes
		bool can_write = ((flush && RLE_buffer.size() > 0) || (RLE_buffer.size() > 511));
		while (can_write)
		{
			write_internal(&(RLE_buffer.front()), 1);
			if (RLE_buffer.front() == 0)
			{
				int count = 0;
				while ((!RLE_buffer.empty()) && (RLE_buffer.front() == 0) && (count < 256))
				{
					count++;
					RLE_buffer.pop_front();
				}
				unsigned char c = (unsigned char)(count - 1);
				write_internal(&c, 1);
			}
			else
			{
				RLE_buffer.pop_front();
			}
			can_write = ((flush && RLE_buffer.size() > 0) || (RLE_buffer.size() > 511));
		}
	}

	void read_arithmetic()
	{
		// check if there is enough data in the stream
		int size = (int)RLE_buffer.size();
		for (int j = 0; j < 1024 - size; j++)
		{
			char sym;
			if (read_internal(&sym, 1) > 0)
			{
				RLE_buffer.push_back(sym);
			}
			else
				RLE_buffer.push_back(0x7f);
		}
	}

	void write_arithmetic()
	{
		while (!RLE_buffer.empty())
		{
			write_internal(&RLE_buffer.front(), 1);
			RLE_buffer.pop_front();
		}
	}

public:
	VolumeFile() : encoder(RLE_buffer), decoder(RLE_buffer), C(256) {
		RLE = false;
		arithmetic = false;
	}
	~VolumeFile() {}

	void setCompressed(bool comp) { compressed = comp; }

	void enableRLE() { RLE = true; }
	void enableArithmetic()
	{
		arithmetic = true;
		for (int i = 0; i < 256; i++) {
			C.install_symbol(i);
		}
		if (writing)
		{
			encoder.start_encode();
		}
		else
		{
			// fill buffer
			read_arithmetic();
			decoder.start_decode();
		}
	}

	void disableRLE()
	{
		if (RLE)
		{
			write_RLE(true);
		}
		RLE = false;
	}
	void disableArithmetic()
	{
		if (arithmetic)
		{
			if (writing)
			{
				encoder.finish_encode();
				write_arithmetic();
			}
			else
			{
				decoder.finish_decode();
			}
		}
		arithmetic = false;
	}

	bool openRead(char *filename)
	{
		writing = false;
#ifdef ZLIB_SUPPORT
		// gzopen supports compressed and uncompressed files
		compressed = true;
		f = gzopen(filename, "rb");
		return (f != NULL);
#else
		compressed = false;
		fp = fopen(filename, "rb");
		return (fp != NULL);
#endif
	}

	bool openWrite(char *filename)
	{
		writing = true;
		compressed = false;
#ifdef ZLIB_SUPPORT
		size_t len = strlen(filename);
		if ((filename[len - 3] == '.') && (filename[len - 2] == 'g') && (filename[len - 1] == 'z')) compressed = true;
		if (compressed)
		{
			f = gzopen(filename, "wb");
			return (f != NULL);
		}
		else
#endif
		{
			fp = fopen(filename, "wb");
			return (fp != NULL);
		}
	}

	void close()
	{
#ifdef ZLIB_SUPPORT
		if (compressed)
		{
			gzclose(f);
		}
		else
#endif
		{
			fclose(fp);
		}
	}

	int read(void *buffer, unsigned int len)
	{
		if (RLE)
		{
			for (unsigned int i = 0; i < len; i++)
			{
				if (RLE_buffer.empty())
				{
					read_RLE();
				}
				((char *)buffer)[i] = RLE_buffer.front();
				RLE_buffer.pop_front();
			}
			return len;
		}
		else if (arithmetic)
		{
			for (unsigned int i = 0; i < len; i++)
			{
				read_arithmetic();
				((unsigned char *)buffer)[i] = (unsigned char)C.decode(decoder);
			}
			return len;
		}
		else
		{
			return read_internal(buffer, len);
		}
	}

	int write(void *buffer, unsigned int len)
	{
		if (RLE)
		{
			for (unsigned int i = 0 ;i < len; i++) RLE_buffer.push_back(((char *)buffer)[i]);
			write_RLE(false);
			return len;
		}
		else if (arithmetic)
		{
			for (unsigned int i = 0; i < len; i++)
			{
				C.encode(encoder, ((unsigned char *)buffer)[i]);
			}
			write_arithmetic();
			return len;
		}
		else
		{
			return write_internal(buffer, len);
		}
	}
};

// no conversion for raw files :/
template <class T>
T *loadRawFile(char *filename, size_t size, float3 &scale, int raw_skip)
{
	VolumeFile fp;

	if (!fp.openRead(filename))
	{
		fprintf(stderr, "Error opening file '%s'\n", filename);
		return 0;
	}

	T *data = (T *)malloc(size);
	size_t read = 0;

	// read chunks of at most 16MB
	while(read < size) read += fp.read(&(((unsigned char *)data)[read]), (unsigned int)std::min((size_t)16777216ull, size - read));

	fp.close();

	return data;
}

template <typename T>
T log_map(T v)
{
	return (T)floor(256.0 * (1.0 - exp(-8.0 * log(2.0) * (double)v / 4095.0)) + 0.5);
}

template<>
float log_map(float v)
{
	return (float)(1.0 - exp(-8.0 * log(2.0) * (double)v));
}

void *loadDatFile(char *filename, cudaExtent &volumeSize, float3 &scale, unsigned int &elementSize, unsigned int &components)
{
	size_t len = strlen(filename);
#ifdef PVM_SUPPORT
	bool pvm;
	if ((filename[len - 3] == 'p') && (filename[len - 2] == 'v') && (filename[len - 1] == 'm')) pvm = true;
	else pvm = false;
#endif

	unsigned char *raw;

#ifdef PVM_SUPPORT
	if (pvm)
	{
		unsigned int w, h, d, c;
		raw = readPVMvolume(filename, &w, &h, &d, &c, &(scale.x), &(scale.y), &(scale.z));
		volumeSize.width = w;
		volumeSize.height = h;
		volumeSize.depth = d;
		if (c == 2)
		{
			components = 1;
			elementSize = 2;
		}
		else
		{
			components = c;
			elementSize = 1;
		}

		if (elementSize == 2)
		{
			// need to swap endian
			for (size_t idx = 0; idx < 2 * volumeSize.width * volumeSize.height * volumeSize.depth; idx += 2)
				std::swap(raw[idx], raw[idx + 1]);

			// clamp to 12 bit
			for (size_t idx = 0; idx < volumeSize.width * volumeSize.height * volumeSize.depth; idx++)
			{
				if (((unsigned short *)raw)[idx] > 4095) ((unsigned short *)raw)[idx] = 0;
			}
		}
		else if (components != 1)
		{
			// organize components from interleaved to non-interleaved
			unsigned char *tmp;
			if (components == 3) tmp = (unsigned char *)malloc(4 * volumeSize.width * volumeSize.height * volumeSize.depth);
			else tmp = (unsigned char *)malloc(components * volumeSize.width * volumeSize.height * volumeSize.depth);
			for (size_t i = 0; i < volumeSize.width * volumeSize.height * volumeSize.depth; i++)
			{
				for (unsigned int c = 0; c < components; c++)
				{
					tmp[i + c * volumeSize.width * volumeSize.height * volumeSize.depth] = raw[components * i + c];
				}
				if (components == 3)
				{
					tmp[i + 3 * volumeSize.width * volumeSize.height * volumeSize.depth] = std::max(std::max(raw[components * i], raw[components * i + 1]), raw[components * i + 2]);
				}
			}
			std::swap(raw, tmp);
			free(tmp);
		}
	}
	else
#endif
	if (((filename[len - 6] == 'd') && (filename[len - 5] == 'a') && (filename[len - 4] == 't')) ||
	    ((filename[len - 3] == 'd') && (filename[len - 2] == 'a') && (filename[len - 1] == 't')))
	{
		size_t read;
		VolumeFile fp;

		if (!fp.openRead(filename))
		{
			fprintf(stderr, "Error opening file '%s'\n", filename);
			return 0;
		}

		unsigned short header[3];
		read = fp.read(header, sizeof(unsigned short) * 3);

		volumeSize.width = header[0];
		volumeSize.height = header[1];
		volumeSize.depth = header[2];

		raw = (unsigned char *)malloc(sizeof(unsigned short) * volumeSize.width * volumeSize.height * volumeSize.depth);
		for (size_t i = 0; i < volumeSize.depth; i++)
		{
			read += fp.read(&(raw[i * sizeof(unsigned short) * volumeSize.width * volumeSize.height]), (unsigned int)(sizeof(unsigned short) * volumeSize.width * volumeSize.height));
		}
		fp.close();

		components = 1;
		elementSize = 2;
	}
	else
	{
		size_t read;
		VolumeFile fp;

		if (!fp.openRead(filename))
		{
			fprintf(stderr, "Error opening file '%s'\n", filename);
			return 0;
		}

		unsigned char fourcc[5];
		unsigned int volumeDim[3];
		unsigned short volumeComp[2];
		float voxelDim[3];
		read = fp.read(fourcc, 4);
		fourcc[4] = '\0';
		if ((fourcc[0] != 'D') || (fourcc[1] != 'D') || (fourcc[2] != 'V') || ((fourcc[3] != '1') && (fourcc[3] != '2') && (fourcc[3] != '3') && (fourcc[3] != '4')))
		{
			fprintf(stderr, "Error opening file '%s'; fourcc = '%s' \n", filename, fourcc);
			exit(-1);
		}
		read += fp.read((unsigned char *)volumeDim, 3 * sizeof(unsigned int));
		volumeSize.width = volumeDim[0];
		volumeSize.height = volumeDim[1];
		volumeSize.depth = volumeDim[2];
		read += fp.read((unsigned char *)volumeComp, 2 * sizeof(unsigned short));
		components = volumeComp[0];
		elementSize = 1;
		if (volumeComp[1] > 8) elementSize = 2;
		int mask = (1 << volumeComp[1]) - 1;
		read += fp.read((unsigned char *)voxelDim, 3 * sizeof(float));
		scale.x = voxelDim[0];
		scale.y = voxelDim[1];
		scale.z = voxelDim[2];
		size_t volume_size = volumeSize.width * volumeSize.height * volumeSize.depth;
		raw = (unsigned char *)malloc(elementSize * components * volume_size);
		if (fourcc[3] == '1')
		{
			// version 1
			for (size_t i = 0; i < volumeSize.depth; i++)
			{
				read += fp.read(&(raw[i * volumeSize.width * volumeSize.height * components * elementSize]), (unsigned int)(volumeSize.width * volumeSize.height * components * elementSize));
			}
		}
		else
		{
			if (fourcc[3] == '3') fp.enableRLE();
			if (fourcc[3] == '4') fp.enableArithmetic();
			unsigned char buffer[4096];
			unsigned int start = 0;
			unsigned int end = 0;

			// version 2
			for (unsigned int c = 0; c < components; c++)
			{
				for (size_t z = 0; z < volumeSize.depth; z += 4)
				{
					for (size_t y = 0; y < volumeSize.height; y += 4)
					{
						for (size_t x = 0; x < volumeSize.width; x += 4)
						{
							while (end <= 2048)
							{
								read += fp.read(&(buffer[end]), 2048);
								end += 2048;
							}
							if (elementSize == 1)
							{
								char tmp[64];
								start += decompressRBUC8x8<char>(&(buffer[start]), tmp);
								int i = 0;
								for (int z0 = 0; z0 < 4; z0++)
								{
									for (int y0 = 0; y0 < 4; y0++)
									{
										for (int x0 = 0; x0 < 4; x0++)
										{
											if ((x + x0 < volumeSize.width) && (y + y0 < volumeSize.height) && (z + z0 < volumeSize.depth))
											{
												raw[(x + x0) + ((y + y0) + ((z + z0) + (c * volumeSize.depth)) * volumeSize.height) * volumeSize.width]  = tmp[i++] + ((mask + 1) >> 1);
											}
											else
											{
												i++;
											}
										}
									}
								}
							}
							else
							{
								short tmp[64];
								start += decompressRBUC8x8<short>(&(buffer[start]), tmp);
								int i = 0;
								for (int z0 = 0; z0 < 4; z0++)
								{
									for (int y0 = 0; y0 < 4; y0++)
									{
										for (int x0 = 0; x0 < 4; x0++)
										{
											if ((x + x0 < volumeSize.width) && (y + y0 < volumeSize.height) && (z + z0 < volumeSize.depth))
											{
												((unsigned short*)raw)[(x + x0) + ((y + y0) + ((z + z0) + (c * volumeSize.depth)) * volumeSize.height) * volumeSize.width] = tmp[i++] + ((mask + 1) >> 1);
											}
											else
											{
												i++;
											}
										}
									}
								}

							}
							if (start >= 2048)
							{
								start -= 2048;
								end -= 2048;
								memcpy(buffer, &(buffer[2048]), 2048);
							}
						}
					}
				}
			}
		}
		if (fourcc[3] == '3') fp.disableRLE();
		if (fourcc[3] == '4') fp.disableArithmetic();
		fp.close();
		for (size_t i = 1; i < volume_size; i++)
		{
			for (unsigned int c = 0; c < components; c++)
			{
				if (elementSize > 1)
				{
					((unsigned short *)raw)[i + c * volume_size] = (((mask + 1) >> 1) + ((unsigned short *)raw)[i + c * volume_size] + ((unsigned short *)raw)[i - 1 + c * volume_size]) & mask;
				}
				else
				{
					raw[i + c * volume_size] = (((mask + 1) >> 1) + raw[i + c * volume_size] + raw[i - 1 + c * volume_size]) & mask;
				}
			}
		}
	}
	volumeSize.depth *= components;
	return raw;
}

void saveDatFile(char *export_name, cudaExtent &volumeSize, float3 &scale, unsigned int &element_size, unsigned int &element_count, void *raw_volume, int volumeType, int export_version)
{
	size_t len = strlen(export_name);
#ifdef PVM_SUPPORT
	bool pvm;
	if ((export_name[len - 3] == 'p') && (export_name[len - 2] == 'v') && (export_name[len - 1] == 'm')) pvm = true;
	else pvm = false;
	if (pvm)
	{
		size_t volume_size = volumeSize.width * volumeSize.height * volumeSize.depth;
		unsigned char *raw = new unsigned char[volume_size * (1ull << volumeType)];
		for (size_t i = 0; i < volume_size; i++)
		{
			if (volumeType < 2)
			{
				size_t idx = i * element_count * element_size;
				for (unsigned int c = 0; c < element_count; c++)
				{
					if (element_size > 1) raw[idx++] = ((unsigned char *)raw_volume)[(i + c * volume_size) * element_size + 1];
					raw[idx++] = ((unsigned char *)raw_volume)[(i + c * volume_size) * element_size];
				}
			}
			else
			{
				size_t idx = i * (1ull << volumeType);
				for (unsigned int c = 0; c < 4; c++)
				{
					if (element_size > 1) raw[idx++] = ((unsigned char *)raw_volume)[i * (1ull << volumeType) + c * element_size + 1];
					raw[idx++] = ((unsigned char *)raw_volume)[i * (1ull << volumeType) + c * element_size];
				}
			}
		}
		writePVMvolume(export_name, raw, (unsigned int)volumeSize.width, (unsigned int)volumeSize.height, (unsigned int)volumeSize.depth, element_count * element_size, scale.x, scale.y, scale.z);
		delete[]raw;
	}
	else
#endif
#ifdef LIBPNG_SUPPORT
	if ((export_name[len - 3] == 'p') && (export_name[len - 2] == 'n') && (export_name[len - 1] == 'g'))
	{
		char *exp_name = new char[len + 100];
		for (unsigned int i = 0; i < len - 4; i++)
		{
			exp_name[i] = export_name[i];
		}
		exp_name[len - 4] = '_';

		PngImage out;
		out.SetWidth((unsigned int)volumeSize.width);
		out.SetHeight((unsigned int)volumeSize.height);
		if (volumeType >= 2) out.SetComponents(4);
		else out.SetComponents(element_count);
		out.SetBitDepth(4 << element_size);

		for (size_t z = 0; z < volumeSize.depth; z++)
		{
			unsigned int pos = (unsigned int)(len - 3);
			unsigned int den = 1;
			while (10 * den <= z) den *= 10;
			while (den > 0)
			{
				exp_name[pos++] = '0' + ((z / den) % 10);
				den /= 10;
			}
			exp_name[pos++] = '.';
			exp_name[pos++] = 'p';
			exp_name[pos++] = 'n';
			exp_name[pos++] = 'g';
			exp_name[pos++] = (char)0;

			if (volumeType == 0)
			{
				size_t vol_size = volumeSize.width * volumeSize.height * volumeSize.depth;
				for (size_t y = 0; y < volumeSize.height; y++)
				{
					for (size_t x = 0; x < volumeSize.width; x++)
					{
						size_t i = x + (y + z * volumeSize.height) * volumeSize.width;
						for (unsigned int c = 0; c < element_count; c++)
						{
							out.VSetValue((unsigned int)x, (unsigned int)y, c, ((unsigned char *)raw_volume)[i + c * vol_size]);
						}
					}
				}
			}
			else if (volumeType == 1)
			{
				size_t vol_size = volumeSize.width * volumeSize.height * volumeSize.depth;
				for (size_t y = 0; y < volumeSize.height; y++)
				{
					for (size_t x = 0; x < volumeSize.width; x++)
					{
						size_t i = x + (y + z * volumeSize.height) * volumeSize.width;
						for (unsigned int c = 0; c < element_count; c++)
						{
							out.VSetValue((unsigned int)x, (unsigned int)y, c, ((unsigned short *)raw_volume)[i + c * vol_size]);
						}
					}
				}
			}
			else if (volumeType == 2)
			{
				for (size_t y = 0; y < volumeSize.height; y++)
				{
					for (size_t x = 0; x < volumeSize.width; x++)
					{
						size_t i = x + (y + z * volumeSize.height) * volumeSize.width;
						for (unsigned int c = 0; c < element_count; c++)
						{
							out.VSetValue((unsigned int)x, (unsigned int)y, c, ((unsigned char *)raw_volume)[i * 4 + c]);
						}
					}
				}
			}
			else if (volumeType == 3)
			{
				for (size_t y = 0; y < volumeSize.height; y++)
				{
					for (size_t x = 0; x < volumeSize.width; x++)
					{
						size_t i = x + (y + z * volumeSize.height) * volumeSize.width;
						for (unsigned int c = 0; c < element_count; c++)
						{
							out.VSetValue((unsigned int)x, (unsigned int)y, c, ((unsigned short *)raw_volume)[i * 4 + c]);
						}
					}
				}
			}
			out.WriteImage(exp_name);
		}

		delete[] exp_name;
	}
	else
#endif
	if ((export_name[len - 3] == 'k') && (export_name[len - 2] == 't') && (export_name[len - 1] == 'x'))
	{
		// write 3D texture for ASTC compression testing
		bool split = false;
		if (volumeSize.width * volumeSize.height * volumeSize.depth * element_size > 65536 * 1024) split = true;
		for (int xx = 0; xx < (split ? 8 : 1); xx++)
		{
			FILE *out;
			if (split)
			{
				char *tmp_name = new char[len + 100];
				unsigned int pos = 0;
				for (unsigned int i = 0; i < len - 4; i++) tmp_name[pos++] = export_name[i];
				tmp_name[pos++] = '_';
				tmp_name[pos++] = '0' + xx;
				tmp_name[pos++] = '.';
				tmp_name[pos++] = 'k';
				tmp_name[pos++] = 't';
				tmp_name[pos++] = 'x';
				tmp_name[pos++] = (char)0;

				out = fopen(tmp_name, "wb");
				delete[] tmp_name;
			}
			else
			{
				out = fopen(export_name, "wb");
			}
			unsigned char FileIdentifier[12] = { 0xAB, 0x4B, 0x54, 0x58, 0x20, 0x31, 0x31, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A };
			uint32_t endianness = 0x04030201u;
			uint32_t glType;
			uint32_t glTypeSize = element_size;
			uint32_t glFormat;
			uint32_t glInternalFormat;
			uint32_t glBaseInternalFormat;
			uint32_t pixelWidth;
			uint32_t pixelHeight;
			uint32_t pixelDepth;
			uint32_t numberOfArrayElements = 0;
			uint32_t numberOfFaces = 1;
			uint32_t numberOfMipmapLevels = 1;
			uint32_t bytesOfKeyValueData = 0;

			uint32_t sx, sy, sz;
			if (split)
			{
				// multiple of 60
				sx = (uint32_t)((volumeSize.width + 59) / 120) * 60;
				sy = (uint32_t)((volumeSize.height + 59) / 120) * 60;
				sz = (uint32_t)((volumeSize.depth + 59) / 120) * 60;
				if ((xx & 1) == 0)
				{
					pixelWidth = sx;
					sx = 0;
				}
				else
				{
					pixelWidth = (uint32_t)volumeSize.width - sx;
				}
				if ((xx & 2) == 0)
				{
					pixelHeight = sy;
					sy = 0;
				}
				else
				{
					pixelHeight = (uint32_t)volumeSize.height - sy;
				}
				if ((xx & 4) == 0)
				{
					pixelDepth = sz;
					sz = 0;
				}
				else
				{
					pixelDepth = (uint32_t)(((volumeSize.depth - sz) + 59) / 60) * 60;
				}
			}
			else
			{
				pixelWidth = (uint32_t)volumeSize.width;
				pixelHeight = (uint32_t)volumeSize.height;
				pixelDepth = (uint32_t)((volumeSize.depth + 59) / 60) * 60;
				sx = sy = sz = 0;
			}
			std::cout << "  --> " << pixelWidth << "x" << pixelHeight << "x" << pixelDepth << " + " << sx << "x" << sy << "x" << sz << std::endl;
			if ((element_count == 1) && (volumeType < 2))
			{
				if (element_size == 1)
				{
					glType = GL_UNSIGNED_BYTE;
					glFormat = GL_RED;
					glInternalFormat = GL_R8;
					glBaseInternalFormat = GL_RED;
				}
				else
				{
					glType = GL_UNSIGNED_SHORT;
					glFormat = GL_RED;
					glInternalFormat = GL_R16;
					glBaseInternalFormat = GL_RED;
				}
			}
			else
			{
				if (element_size == 1)
				{
					glType = GL_UNSIGNED_BYTE;
					glFormat = GL_RGBA;
					glInternalFormat = GL_RGBA8;
					glBaseInternalFormat = GL_RGBA;
				}
				else
				{
					glType = GL_UNSIGNED_SHORT;
					glFormat = GL_RGBA;
					glInternalFormat = GL_RGBA16;
					glBaseInternalFormat = GL_RGBA;
				}
			}
			fwrite(FileIdentifier, sizeof(unsigned char), 12, out);
			fwrite(&endianness, sizeof(uint32_t), 1, out);
			fwrite(&glType, sizeof(uint32_t), 1, out);
			fwrite(&glTypeSize, sizeof(uint32_t), 1, out);
			fwrite(&glFormat, sizeof(uint32_t), 1, out);
			fwrite(&glInternalFormat, sizeof(uint32_t), 1, out);
			fwrite(&glBaseInternalFormat, sizeof(uint32_t), 1, out);
			fwrite(&pixelWidth, sizeof(uint32_t), 1, out);
			fwrite(&pixelHeight, sizeof(uint32_t), 1, out);
			fwrite(&pixelDepth, sizeof(uint32_t), 1, out);
			fwrite(&numberOfArrayElements, sizeof(uint32_t), 1, out);
			fwrite(&numberOfFaces, sizeof(uint32_t), 1, out);
			fwrite(&numberOfMipmapLevels, sizeof(uint32_t), 1, out);
			fwrite(&bytesOfKeyValueData, sizeof(uint32_t), 1, out);

			uint32_t imageSize;
			if (volumeType < 2)
			{
				imageSize = pixelWidth * pixelHeight * pixelDepth * element_size * element_count;
			}
			else
			{
				imageSize = pixelWidth * pixelHeight * pixelDepth * element_size * 4;
			}
			fwrite(&imageSize, sizeof(uint32_t), 1, out);

			unsigned int dd = 0;

			if (volumeType == 0)
			{
				size_t vol_size = volumeSize.width * volumeSize.height * volumeSize.depth;
				for (size_t z = 0; z < pixelDepth; z++)
				{
					for (size_t y = 0; y < pixelHeight; y++)
					{
						for (size_t x = 0; x < pixelWidth; x++)
						{
							size_t i = (sx + x) + ((sy + y) + (sz + z) * volumeSize.height) * volumeSize.width;
							if (sz + z >= volumeSize.depth)
							{
								for (unsigned int c = 0; c < element_count; c++)
								{
									fwrite(&dd, sizeof(unsigned char), 1, out);
								}
							}
							else
							{
								for (unsigned int c = 0; c < element_count; c++)
								{
									fwrite(&(((unsigned char *)raw_volume)[i + c * vol_size]), sizeof(unsigned char), 1, out);
								}
							}
						}
					}
				}
			}
			else if (volumeType == 1)
			{
				size_t vol_size = volumeSize.width * volumeSize.height * volumeSize.depth;
				for (size_t z = 0; z < pixelDepth; z++)
				{
					for (size_t y = 0; y < pixelHeight; y++)
					{
						for (size_t x = 0; x < pixelWidth; x++)
						{
							size_t i = (sx + x) + ((sy + y) + (sz + z) * volumeSize.height) * volumeSize.width;
							if (sz + z >= volumeSize.depth)
							{
								for (unsigned int c = 0; c < element_count; c++)
								{
									fwrite(&dd, sizeof(unsigned short), 1, out);
								}
							}
							else
							{
								for (unsigned int c = 0; c < element_count; c++)
								{
									fwrite(&(((unsigned short *)raw_volume)[i + c * vol_size]), sizeof(unsigned short), 1, out);
								}
							}
						}
					}
				}
			}
			else if (volumeType == 2)
			{
				for (size_t z = 0; z < pixelDepth; z++)
				{
					for (size_t y = 0; y < pixelHeight; y++)
					{
						for (size_t x = 0; x < pixelWidth; x++)
						{
							size_t i = (sx + x) + ((sy + y) + (sz + z) * volumeSize.height) * volumeSize.width;
							if (sz + z >= volumeSize.depth)
							{
								for (unsigned int c = 0; c < 4; c++)
								{
									fwrite(&dd, sizeof(unsigned char), 1, out);
								}
							}
							else
							{
								for (unsigned int c = 0; c < 4; c++)
								{
									fwrite(&(((unsigned char *)raw_volume)[i * 4 + c]), sizeof(unsigned char), 1, out);
								}
							}
						}
					}
				}
			}
			else if (volumeType == 3)
			{
				for (size_t z = 0; z < pixelDepth; z++)
				{
					for (size_t y = 0; y < pixelHeight; y++)
					{
						for (size_t x = 0; x < pixelWidth; x++)
						{
							size_t i = (sx + x) + ((sy + y) + (sz + z) * volumeSize.height) * volumeSize.width;
							if (sz + z >= volumeSize.depth)
							{
								for (unsigned int c = 0; c < 4; c++)
								{
									fwrite(&dd, sizeof(unsigned short), 1, out);
								}
							}
							else
							{
								for (unsigned int c = 0; c < 4; c++)
								{
									fwrite(&(((unsigned short *)raw_volume)[i * 4 + c]), sizeof(unsigned short), 1, out);
								}
							}
						}
					}
				}
			}

			fclose(out);
		}
	}
	else if ((export_name[len - 3] == 'f') && (export_name[len - 2] == '3') && (export_name[len - 1] == '2'))
	{
		// write float array for ZFP testing
		size_t total;
		float *dat;
		FILE *out = fopen(export_name, "wb");
		if (volumeType < 2)
		{
			total = volumeSize.width * volumeSize.height * volumeSize.depth * element_count;
		}
		else
		{
			total = volumeSize.width * volumeSize.height * volumeSize.depth * 4;
		}
		dat = new float[total];
		if (volumeType == 0)
		{
			for (size_t i = 0; i < volumeSize.width * volumeSize.height * volumeSize.depth * element_count; i++)
				dat[i] = (float)(((unsigned char *)raw_volume)[i]) / 255.0f;
		}
		else if (volumeType == 1)
		{
			for (size_t i = 0; i < volumeSize.width * volumeSize.height * volumeSize.depth * element_count; i++)
				dat[i] = (float)(((unsigned short *)raw_volume)[i]) / 4095.0f;
		}
		else if (volumeType == 2)
		{
			for (size_t i = 0; i < volumeSize.width * volumeSize.height * volumeSize.depth; i++)
			for (size_t j = 0; j < 4; j++)
				dat[i + j * volumeSize.width * volumeSize.height * volumeSize.depth] = (float)(((unsigned char *)raw_volume)[i * 4 + j]) / 255.0f;
		}
		else
		{
			for (size_t i = 0; i < volumeSize.width * volumeSize.height * volumeSize.depth; i++)
			for (size_t j = 0; j < 4; j++)
				dat[i + j * volumeSize.width * volumeSize.height * volumeSize.depth] = (float)(((unsigned short *)raw_volume)[i * 4 + j]) / 4095.0f;
		}

		fwrite(dat, sizeof(float), total, out);
		fclose(out);
		delete[] dat;
	}
	else
	{
		// raw volume export
		VolumeFile out;
		out.openWrite(export_name);
		// 8-bit is 'raw', 16-bit is 'dat'
		if (((export_name[len - 6] == 'd') && (export_name[len - 5] == 'a') && (export_name[len - 4] == 't')) ||
			((export_name[len - 6] == 'r') && (export_name[len - 5] == 'a') && (export_name[len - 4] == 'w')) ||
			((export_name[len - 3] == 'd') && (export_name[len - 2] == 'a') && (export_name[len - 1] == 't')) ||
			((export_name[len - 3] == 'r') && (export_name[len - 2] == 'a') && (export_name[len - 1] == 'w')))
		{
			unsigned char *raw_copy = (unsigned char *)raw_volume;
			if (volumeType >= 2)
			{
				element_count = 4;
				raw_copy = new unsigned char[volumeSize.width * volumeSize.height * volumeSize.depth * element_count * element_size];
				size_t vol_size = volumeSize.width * volumeSize.height * volumeSize.depth;
				for (size_t i = 0; i < vol_size; i++)
				{
					for (size_t c = 0; c < element_count; c++)
					{
						for (size_t e = 0; e < element_size; e++)
						{
							raw_copy[i * element_size + c * vol_size * element_size + e] = ((unsigned char *)raw_volume)[i * element_size * element_count + c * element_size + e];
						}
					}
				}
			}
			if (element_size > 1)
			{
				unsigned short header[3];
				header[0] = (unsigned short)volumeSize.width;
				header[1] = (unsigned short)volumeSize.height;
				header[2] = (unsigned short)(volumeSize.depth * element_count);
				out.write(header, (unsigned int)(sizeof(unsigned short)* 3));
			}
			out.write(raw_copy, (unsigned int)(volumeSize.width * volumeSize.height * volumeSize.depth * element_count * element_size));
			if (volumeType >= 2)
			{
				delete[] raw_copy;
				element_count = 1;
			}
		}
		else
		{
			// new volume format
			char fourcc[] = "DDVx";
			// uncompressed
			if (export_version == 1) fourcc[3] = '1';
			// RBUC
			else if (export_version == 2) fourcc[3] = '2';
			// RBUC + RLE0
			else if (export_version == 3) fourcc[3] = '3';
			// RBUS + arithmetic
			else fourcc[3] = '4';

			if (volumeType >= 2)
			{
				element_count = 4;
			}

			unsigned int volumeDim[3] = { (unsigned int)volumeSize.width, (unsigned int)volumeSize.height, (unsigned int)volumeSize.depth };
			unsigned short volumeComp[2] = { element_count, element_size };
			float voxelDim[3] = { scale.x, scale.y, scale.z };
			size_t volume_size = volumeSize.width * volumeSize.height * volumeSize.depth;
			unsigned char *raw = new unsigned char[volume_size * element_count * element_size];
			int mask = 0;
			if (volumeType < 2)
			{
				for (size_t i = 0; i < volume_size; i++)
				{
					for (unsigned int c = 0; c < element_count; c++)
					{
						if (element_size > 1)
						{
							((unsigned short *)raw)[i + c * volume_size] = ((unsigned short *)raw_volume)[i + c * volume_size];
							mask = std::max(mask, (int)((unsigned short *)raw_volume)[i + c * volume_size]);
						}
						else
						{
							raw[i + c * volume_size] = ((unsigned char *)raw_volume)[i + c * volume_size];
							mask = std::max(mask, (int)((unsigned char *)raw_volume)[i + c * volume_size]);
						}
					}
				}
			}
			else
			{
				for (size_t i = 0; i < volume_size; i++)
				{
					for (unsigned int c = 0; c < element_count; c++)
					{
						if (element_size > 1)
						{
							((unsigned short *)raw)[i + c * volume_size] = ((unsigned short *)raw_volume)[i * element_count + c];
							mask = std::max(mask, (int)((unsigned short *)raw_volume)[i * element_count + c]);
						}
						else
						{
							raw[i + c * volume_size] = ((unsigned char *)raw_volume)[i * element_count + c];
							mask = std::max(mask, (int)((unsigned char *)raw_volume)[i * element_count + c]);
						}
					}
				}
			}
			if (volumeComp[1] == 2) volumeComp[1] = 9;
			while ((1 << volumeComp[1]) <= mask) volumeComp[1] <<= 1;
			mask = (1 << volumeComp[1]) - 1;
			out.write(fourcc, 4);
			out.write((unsigned char *)volumeDim, 3 * sizeof(unsigned int));
			out.write((unsigned char *)volumeComp, 2 * sizeof(unsigned short));
			out.write((unsigned char *)voxelDim, 3 * sizeof(float));
			for (size_t i = volume_size - 1; i > 0; i--)
			{
				for (unsigned int c = 0; c < element_count; c++)
				{
					if (element_size > 1)
					{
						((unsigned short *)raw)[i + c * volume_size] = (((mask + 1) >> 1) + ((unsigned short *)raw)[i + c * volume_size] - ((unsigned short *)raw)[i - 1 + c * volume_size]) & mask;
					}
					else
					{
						raw[i + c * volume_size] = (((mask + 1) >> 1) + raw[i + c * volume_size] - raw[i - 1 + c * volume_size]) & mask;
					}
				}
			}
			if (fourcc[3] != '1')
			{
				if (fourcc[3] == '3') out.enableRLE();
				if (fourcc[3] == '4') out.enableArithmetic();
				if (element_size == 1)
				{
					for (unsigned int c = 0; c < element_count; c++)
					{
						for (size_t z = 0; z < volumeSize.depth; z += 4)
						{
							for (size_t y = 0; y < volumeSize.height; y += 4)
							{
								for (size_t x = 0; x < volumeSize.width; x += 4)
								{
									unsigned char comp[64 + 1];
									unsigned int comp_bytes = 0;
									char tmp[64];
									int i = 0;
									for (int z0 = 0; z0 < 4; z0++)
									{
										for (int y0 = 0; y0 < 4; y0++)
										{
											for (int x0 = 0; x0 < 4; x0++)
											{
												if ((x + x0 < volumeSize.width) && (y + y0 < volumeSize.height) && (z + z0 < volumeSize.depth))
												{
													tmp[i++] = raw[(x + x0) + ((y + y0) + ((z + z0) + (c * volumeSize.depth)) * volumeSize.height) * volumeSize.width] - ((mask + 1) >> 1);
												}
												else
												{
													tmp[i++] = 0;
												}
											}
										}
									}
									comp_bytes = compressRBUC8x8<char>(tmp, comp);
									out.write(comp, comp_bytes);
								}
							}
						}
					}
				}
				else
				{
					for (unsigned int c = 0; c < element_count; c++)
					{
						for (size_t z = 0; z < volumeSize.depth; z += 4)
						{
							for (size_t y = 0; y < volumeSize.height; y += 4)
							{
								for (size_t x = 0; x < volumeSize.width; x += 4)
								{
									unsigned char comp[64 + 1];
									unsigned int comp_bytes = 0;
									short tmp[64];
									int i = 0;
									for (int z0 = 0; z0 < 4; z0++)
									{
										for (int y0 = 0; y0 < 4; y0++)
										{
											for (int x0 = 0; x0 < 4; x0++)
											{
												if ((x + x0 < volumeSize.width) && (y + y0 < volumeSize.height) && (z + z0 < volumeSize.depth))
												{
													tmp[i++] = ((unsigned short*)raw)[(x + x0) + ((y + y0) + ((z + z0) + (c * volumeSize.depth)) * volumeSize.height) * volumeSize.width] - ((mask + 1) >> 1);
												}
												else
												{
													tmp[i++] = 0;
												}
											}
										}
									}
									comp_bytes = compressRBUC8x8<short>(tmp, comp);
									out.write(comp, comp_bytes);
								}
							}
						}
					}
				}
				if (fourcc[3] == '3') out.disableRLE();
				if (fourcc[3] == '4') out.disableArithmetic();
			}
			else
			{
				// version 1 is uncompressed
				for (size_t i = 0; i < volumeSize.depth; i++)
				{
					out.write(&(raw[i * volumeSize.width * volumeSize.height * element_count * element_size]), (unsigned int)(volumeSize.width * volumeSize.height * element_count * element_size));
				}
			}
			delete[] raw;
			if (volumeType >= 2)
			{
				element_count = 1;
			}
		}
		out.close();
	}
}

template <class T>
void denoiseVolume(T *vol, cudaExtent &volumeSize, int denoise)
{
	if (denoise == 1)
	{
#pragma omp parallel for
		for (int z = 0; z < volumeSize.depth; z++)
		{
			for (size_t y = 0; y < volumeSize.height; y++)
			{
				T min_data = T(0xffffffff);
				for (size_t x = 0; x < volumeSize.width; x += volumeSize.width - 1)
				{
					min_data = std::min(min_data, vol[x + (y + z * volumeSize.height) * volumeSize.width]);
				}
				for (size_t x = 0; x < volumeSize.width; x++)
				{
					if (vol[x + (y + z * volumeSize.height) * volumeSize.width] < min_data)
						vol[x + (y + z * volumeSize.height) * volumeSize.width] = 0;
					else
						vol[x + (y + z * volumeSize.height) * volumeSize.width] -= min_data;
				}
			}
		}
	}
	if (denoise == 2)
	{
#pragma omp parallel for
		for (int z = 0; z < volumeSize.depth; z++)
		{
			for (size_t x = 0; x < volumeSize.width; x++)
			{
				T min_data = T(0xffffffff);
				for (size_t y = 0; y < volumeSize.height; y += volumeSize.height - 1)
				{
					min_data = std::min(min_data, vol[x + (y + z * volumeSize.height) * volumeSize.width]);
				}
				for (size_t y = 0; y < volumeSize.height; y++)
				{
					if (vol[x + (y + z * volumeSize.height) * volumeSize.width] < min_data)
						vol[x + (y + z * volumeSize.height) * volumeSize.width] = 0;
					else
						vol[x + (y + z * volumeSize.height) * volumeSize.width] -= min_data;
				}
			}
		}
	}
	if (denoise == 3)
	{
#pragma omp parallel for
		for (int y = 0; y < volumeSize.height; y++)
		{
			for (size_t x = 0; x < volumeSize.width; x++)
			{
				T min_data = T(0xffffffff);
				for (size_t z = 0; z < volumeSize.depth; z += volumeSize.depth - 1)
				{
					min_data = std::min(min_data, vol[x + (y + z * volumeSize.height) * volumeSize.width]);
				}
				for (size_t z = 0; z < volumeSize.depth; z++)
				{
					if (vol[x + (y + z * volumeSize.height) * volumeSize.width] < min_data)
						vol[x + (y + z * volumeSize.height) * volumeSize.width] = 0;
					else
						vol[x + (y + z * volumeSize.height) * volumeSize.width] -= min_data;
				}
			}
		}
	}

	if (denoise == 4)
	{
#pragma omp parallel for
		for (int z = 0; z < volumeSize.depth; z++)
		{
			T layer_max;
			bool first = true;
			for (size_t y = 0; y < volumeSize.height; y++)
			{
				for (size_t x = 0; x < volumeSize.width; x++)
				{
					float rx = 2.0f*((float)x / (volumeSize.width - 1.0f)) - 1.0f;
					float ry = 2.0f*((float)y / (volumeSize.height - 1.0f)) - 1.0f;
					float r = rx * rx + ry * ry;
					if ((r < 0.95f) && ((rx * rx > 0.6f) || (ry * ry > 0.6f)))
					{
						if (first)
						{
							layer_max = vol[x + (y + z * volumeSize.height) * volumeSize.width];
							first = false;
						}
						else
						{
							layer_max = std::max(layer_max, vol[x + (y + z * volumeSize.height) * volumeSize.width]);
						}
					}
				}
			}
			if (layer_max > 0)
			{
				for (size_t y = 0; y < volumeSize.height; y++)
				{
					for (size_t x = 0; x < volumeSize.width; x++)
					{
						if (vol[x + (y + z * volumeSize.height) * volumeSize.width] < layer_max)
							vol[x + (y + z * volumeSize.height) * volumeSize.width] = 0;
					}
				}
			}
		}
	}
}

template <class T>
void linearWaveletTransform(T *vol, cudaExtent &volumeSize, int dist, int axis)
{
	for (unsigned int z = ((axis == 2) ? 1 : 0) * dist; z < volumeSize.depth; z += ((axis == 2) ? 2 : 1) * dist)
	{
		for (unsigned int y = ((axis == 1) ? 1 : 0) * dist; y < volumeSize.height; y += ((axis == 1) ? 2 : 1) * dist)
		{
			for (unsigned int x = ((axis == 0) ? 1 : 0) * dist; x < volumeSize.width; x += ((axis == 0) ? 2 : 1) * dist)
			{
				size_t s = x + (y + z * volumeSize.height) * volumeSize.width;
				size_t s0, s1;
				switch (axis)
				{
					case 0:
						if (x > 0) s0 = s - dist; else s0 = s + dist;
						if (x + dist < volumeSize.width) s1 = s + dist; else s1 = s - dist;
						break;
					case 1:
						if (y > 0) s0 = s - volumeSize.width * dist; else s0 = s + volumeSize.width * dist;
						if (y + dist < volumeSize.height) s1 = s + volumeSize.width * dist; else s1 = s - volumeSize.width * dist;
						break;
					default:
						if (z > 0) s0 = s - volumeSize.width * volumeSize.height * dist; else s0 = s + volumeSize.width * volumeSize.height * dist;
						if (z + dist < volumeSize.depth) s1 = s + volumeSize.width * volumeSize.height * dist; else s1 = s - volumeSize.width * volumeSize.height * dist;
				}
				vol[s] -= vol[s0];// (vol[s0] + vol[s1]) / 2.0;
			}
		}
	}
	for (unsigned int z = 0; z < volumeSize.depth; z += ((axis == 2) ? 2 : 1) * dist)
	{
		for (unsigned int y = 0; y < volumeSize.height; y += ((axis == 1) ? 2 : 1) * dist)
		{
			for (unsigned int x = 0; x < volumeSize.width; x += ((axis == 0) ? 2 : 1) * dist)
			{
				size_t s = x + (y + z * volumeSize.height) * volumeSize.width;
				size_t s0, s1;
				switch (axis)
				{
				case 0:
					if (x > 0) s0 = s - dist; else s0 = s + dist;
					if (x + dist < volumeSize.width) s1 = s + dist; else s1 = s - dist;
					break;
				case 1:
					if (y > 0) s0 = s - volumeSize.width * dist; else s0 = s + volumeSize.width * dist;
					if (y + dist < volumeSize.height) s1 = s + volumeSize.width * dist; else s1 = s - volumeSize.width * dist;
					break;
				default:
					if (z > 0) s0 = s - volumeSize.width * volumeSize.height * dist; else s0 = s + volumeSize.width * volumeSize.height * dist;
					if (z + dist < volumeSize.depth) s1 = s + volumeSize.width * volumeSize.height * dist; else s1 = s - volumeSize.width * volumeSize.height * dist;
				}
				vol[s] += vol[s1] / 2.0;// (vol[s0] + vol[s1]) / 4.0;
			}
		}
	}
} 

template <class T>
void linearInverseTransform(T *vol, cudaExtent &volumeSize, int dist, int axis)
{
	for (unsigned int z = 0; z < volumeSize.depth; z += ((axis == 2) ? 2 : 1) * dist)
	{
		for (unsigned int y = 0; y < volumeSize.height; y += ((axis == 1) ? 2 : 1) * dist)
		{
			for (unsigned int x = 0; x < volumeSize.width; x += ((axis == 0) ? 2 : 1) * dist)
			{
				size_t s = x + (y + z * volumeSize.height) * volumeSize.width;
				size_t s0, s1;
				switch (axis)
				{
				case 0:
					if (x > 0) s0 = s - dist; else s0 = s + dist;
					if (x + dist < volumeSize.width) s1 = s + dist; else s1 = s - dist;
					break;
				case 1:
					if (y > 0) s0 = s - volumeSize.width * dist; else s0 = s + volumeSize.width * dist;
					if (y + dist < volumeSize.height) s1 = s + volumeSize.width * dist; else s1 = s - volumeSize.width * dist;
					break;
				default:
					if (z > 0) s0 = s - volumeSize.width * volumeSize.height * dist; else s0 = s + volumeSize.width * volumeSize.height * dist;
					if (z + dist < volumeSize.depth) s1 = s + volumeSize.width * volumeSize.height * dist; else s1 = s - volumeSize.width * volumeSize.height * dist;
				}
				vol[s] -= vol[s1] / 2.0;
			}
		}
	}
	for (unsigned int z = ((axis == 2) ? 1 : 0) * dist; z < volumeSize.depth; z += ((axis == 2) ? 2 : 1) * dist)
	{
		for (unsigned int y = ((axis == 1) ? 1 : 0) * dist; y < volumeSize.height; y += ((axis == 1) ? 2 : 1) * dist)
		{
			for (unsigned int x = ((axis == 0) ? 1 : 0) * dist; x < volumeSize.width; x += ((axis == 0) ? 2 : 1) * dist)
			{
				size_t s = x + (y + z * volumeSize.height) * volumeSize.width;
				size_t s0, s1;
				switch (axis)
				{
				case 0:
					if (x > 0) s0 = s - dist; else s0 = s + dist;
					if (x + dist < volumeSize.width) s1 = s + dist; else s1 = s - dist;
					break;
				case 1:
					if (y > 0) s0 = s - volumeSize.width * dist; else s0 = s + volumeSize.width * dist;
					if (y + dist < volumeSize.height) s1 = s + volumeSize.width * dist; else s1 = s - volumeSize.width * dist;
					break;
				default:
					if (z > 0) s0 = s - volumeSize.width * volumeSize.height * dist; else s0 = s + volumeSize.width * volumeSize.height * dist;
					if (z + dist < volumeSize.depth) s1 = s + volumeSize.width * volumeSize.height * dist; else s1 = s - volumeSize.width * volumeSize.height * dist;
				}
				vol[s] += vol[s0];
			}
		}
	}
}

float quantizeMaximum(unsigned char *vol) { return 255.0f; }
float quantizeMaximum(unsigned short *vol) { return 4095.0f; }
float quantizeMaximum(uchar4 *vol) { return 255.0f; }
float quantizeMaximum(ushort4 *vol) { return 4095.0f; }

template <class T> T zero();

template<> uchar zero() { return 0; }
template<> unsigned short zero() { return 0; }
template<> uint zero() { return 0; }
template<> unsigned long long zero() { return 0; }
template<> uchar4 zero() { return make_uchar4(0, 0, 0, 0); }
template<> ushort4 zero() { return make_ushort4(0, 0, 0, 0); }
template<> uint4 zero() { return make_uint4(0, 0, 0, 0); }
template<> ulonglong4 zero() { return make_ulonglong4(0, 0, 0, 0); }

#ifndef WIN32
#define __forceinline inline __attribute__((always_inline))
#endif

template<class A, class D>
class VolumeBlock 
{
public:
	D value[64];
	unsigned short hash;
	unsigned int count;
	unsigned int last;

	A lastDist;
	A squared;
	double zeroDist;

	__forceinline A dot(const D &a, const D &b);
	__forceinline bool equal(const D &a, const D &b);
	__forceinline void hashing(const D &a);

	__forceinline bool operator==(VolumeBlock<A, D> &a)
	{
		if (a.hash != hash) return false;
		for (unsigned int i = 0; i < 64; i++) if (!equal(value[i], a.value[i])) return false;
		return true;
	}

	__forceinline void calcHash()
	{
		hash = 0;
		for (unsigned int i = 0; i < 64; i++) hashing(value[i]);
		squared = A(0);
		for (unsigned int i = 0; i < 64; i++) squared += dot(value[i], value[i]);
		zeroDist = sqrt((double)squared);
	}

	__forceinline A dist(const VolumeBlock<A, D> &v)
	{
		A dd = A(0);
		for (unsigned int i = 0; i < 64; i++)
		{
			dd += dot(value[i], v.value[i]);
		}
		A d = squared + v.squared;
		d -= dd;
		d -= dd;
		return d;
	}

	__forceinline A dist()
	{
		return squared;
	}

	__forceinline bool dist(VolumeBlock<A, D> &v, A min_dist, A &d)
	{
		if ((zeroDist - v.zeroDist)*(zeroDist - v.zeroDist) > min_dist) return false;
		A dd = A(0);
		for (unsigned int i = 0; i < 64; i++)
		{
			dd += dot(value[i], v.value[i]);
		}
		d = squared + v.squared;
		d -= dd;
		d -= dd;
		if (d < min_dist)
		{
			return true;
		}
		return false;
	}
};

#include <omp.h>

template <class A, class D, class O>
class CenterBlock
{
private:
	omp_lock_t lock;
public:
	O value[64];
	D sum[64];
	uint weight;
	bool changedCenter;
	bool changedMember;
	A squared;
	double zeroDist;

	CenterBlock() { omp_init_lock(&lock); }

	__forceinline A dot(const O &a, const O &b);
	__forceinline A dot2(const O &a, const D &b);
	__forceinline void add(D &s, const O &v, const int c);
	__forceinline void convert(O &v, const D &s);
	__forceinline void assign(O &v, const D &s);

	__forceinline A dist(VolumeBlock<A, O> &v)
	{
		A dd = A(0);
		for (unsigned int i = 0; i < 64; i++)
		{
			dd += dot(value[i], v.value[i]);
		}
		A d = squared + v.squared;
 		d -= dd;
		d -= dd;
		return d;
	}

	__forceinline bool dist(VolumeBlock<A, O> &v, A &min_dist, bool eq)
	{
		if ((zeroDist - v.zeroDist)*(zeroDist - v.zeroDist) > min_dist) return false;
		A dd = A(0);
		for (unsigned int i = 0; i < 64; i++)
		{
			dd += dot(value[i], v.value[i]);
		}
		A d = squared + v.squared;
		d -= dd;
		d -= dd;
		if ((d < min_dist) || ((d == min_dist) && eq))
		{
			min_dist = d;
			return true;
		}
		return false;
	}

	__forceinline void add(const VolumeBlock<A, O> &v)
	{
		omp_set_lock(&lock);
		for (unsigned int i = 0; i < 64; i++)
		{
			add(sum[i], v.value[i], (int)v.count);
		}
		weight += v.count;
		omp_unset_lock(&lock);
	}

	__forceinline void sub(const VolumeBlock<A, O> &v)
	{
		omp_set_lock(&lock);
		for (unsigned int i = 0; i < 64; i++)
		{
			add(sum[i], v.value[i], -((int)v.count));
		}
		weight -= v.count;
		omp_unset_lock(&lock);
	}

	__forceinline void update()
	{
		changedCenter = false;
		if (!changedMember) return;
		changedMember = false;
		if (weight > 0)
		{
			squared = A(0);
			for (unsigned int i = 0; i < 64; i++)
			{
				convert(value[i], sum[i]);
			}
			zeroDist = sqrt((double)squared);
		}
	}
	__forceinline CenterBlock<A, D, O> &operator=(VolumeBlock<A, O> &a)
	{
		for (unsigned int i = 0; i < 64; i++)
		{
			//assign(value[i], a.value[i]);
			value[i] = a.value[i];
		}
		for (unsigned int i = 0; i < 64; i++)
		{
			sum[i] = zero<D>();
		}
		weight = 0;
		changedCenter = true;
		changedMember = true;
		squared = a.squared;
		zeroDist = a.zeroDist;
		return *this;
	}
};

template<> __forceinline unsigned long long VolumeBlock<unsigned long long, uchar>::dot(const uchar &a, const uchar &b) { return (unsigned long long)((unsigned int)a * (unsigned int)b); }
template<> __forceinline unsigned long long VolumeBlock<unsigned long long, unsigned short>::dot(const unsigned short &a, const unsigned short &b) { return (unsigned long long)((unsigned int)a * (unsigned int)b); }
template<> __forceinline unsigned long long VolumeBlock<unsigned long long, uchar4>::dot(const uchar4 &a, const uchar4 &b) { return (unsigned long long)((unsigned int)a.x * (unsigned int)b.x + (unsigned int)a.y * (unsigned int)b.y + (unsigned int)a.z * (unsigned int)b.z + (unsigned int)a.w * (unsigned int)b.w); }
template<> __forceinline unsigned long long VolumeBlock<unsigned long long, ushort4>::dot(const ushort4 &a, const ushort4 &b) { return (unsigned long long)((unsigned int)a.x * (unsigned int)b.x + (unsigned int)a.y * (unsigned int)b.y + (unsigned int)a.z * (unsigned int)b.z + (unsigned int)a.w * (unsigned int)b.w); }

template<> __forceinline bool VolumeBlock<unsigned long long, uchar>::equal(const uchar &a, const uchar &b) { return a == b; }
template<> __forceinline bool VolumeBlock<unsigned long long, unsigned short>::equal(const unsigned short &a, const unsigned short &b) { return a == b; }
template<> __forceinline bool VolumeBlock<unsigned long long, uchar4>::equal(const uchar4 &a, const uchar4 &b) { return (a.x == b.x) && (a.y == b.y) && (a.z == b.z) && (a.w == b.w); }
template<> __forceinline bool VolumeBlock<unsigned long long, ushort4>::equal(const ushort4 &a, const ushort4 &b) { return (a.x == b.x) && (a.y == b.y) && (a.z == b.z) && (a.w == b.w); }

template<> __forceinline void VolumeBlock<unsigned long long, uchar>::hashing(const uchar &a) { hash = (hash << 1) ^ (hash >> 15) ^ a; }
template<> __forceinline void VolumeBlock<unsigned long long, unsigned short>::hashing(const unsigned short &a) { hash = (hash << 1) ^ (hash >> 15) ^ a; }
template<> __forceinline void VolumeBlock<unsigned long long, uchar4>::hashing(const uchar4 &a) { hash = (hash << 1) ^ (hash >> 15) ^ a.x; hash = (hash << 1) ^ (hash >> 15) ^ a.y; hash = (hash << 1) ^ (hash >> 15) ^ a.z; hash = (hash << 1) ^ (hash >> 15) ^ a.w; }
template<> __forceinline void VolumeBlock<unsigned long long, ushort4>::hashing(const ushort4 &a) { hash = (hash << 1) ^ (hash >> 15) ^ a.x; hash = (hash << 1) ^ (hash >> 15) ^ a.y; hash = (hash << 1) ^ (hash >> 15) ^ a.z; hash = (hash << 1) ^ (hash >> 15) ^ a.w; }

template<> __forceinline unsigned long long CenterBlock<unsigned long long, uint, uchar>::dot2(const uchar &a, const uint &b) { return (unsigned long long)a * (unsigned long long)b; }
template<> __forceinline unsigned long long CenterBlock<unsigned long long, unsigned long long, unsigned short>::dot2(const unsigned short &a, const unsigned long long &b) { return (unsigned long long)a * b; }
template<> __forceinline unsigned long long CenterBlock<unsigned long long, uint4, uchar4>::dot2(const uchar4 &a, const uint4 &b) { return (unsigned long long)a.x * (unsigned long long)b.x + (unsigned long long)a.y * (unsigned long long)b.y + (unsigned long long)a.z * (unsigned long long)b.z + (unsigned long long)a.w * (unsigned long long)b.w; }
template<> __forceinline unsigned long long CenterBlock<unsigned long long, ulonglong4, ushort4>::dot2(const ushort4 &a, const ulonglong4 &b) { return (unsigned long long)a.x * b.x + (unsigned long long)a.y * b.y + (unsigned long long)a.z * b.z + (unsigned long long)a.w * b.w; }

template<> __forceinline unsigned long long CenterBlock<unsigned long long, uint, uchar>::dot(const uchar &a, const uchar &b) { return (unsigned long long)((unsigned int)a * (unsigned int)b); }
template<> __forceinline unsigned long long CenterBlock<unsigned long long, unsigned long long, unsigned short>::dot(const unsigned short &a, const unsigned short &b) { return (unsigned long long)((unsigned int)a * (unsigned int)b); }
template<> __forceinline unsigned long long CenterBlock<unsigned long long, uint4, uchar4>::dot(const uchar4 &a, const uchar4 &b) { return (unsigned long long)((unsigned int)a.x * (unsigned int)b.x + (unsigned int)a.y * (unsigned int)b.y + (unsigned int)a.z * (unsigned int)b.z + (unsigned int)a.w * (unsigned int)b.w); }
template<> __forceinline unsigned long long CenterBlock<unsigned long long, ulonglong4, ushort4>::dot(const ushort4 &a, const ushort4 &b) { return (unsigned long long)((unsigned int)a.x * (unsigned int)b.x + (unsigned int)a.y * (unsigned int)b.y + (unsigned int)a.z * (unsigned int)b.z + (unsigned int)a.w * (unsigned int)b.w); }

template<> __forceinline void CenterBlock<unsigned long long, uint, uchar>::add(uint &s, const uchar &v, const int c) { s += (int)v * c; }
template<> __forceinline void CenterBlock<unsigned long long, unsigned long long, unsigned short>::add(unsigned long long &s, const unsigned short &v, const int c) { s += (long long)v * (long long)c; }
template<> __forceinline void CenterBlock<unsigned long long, uint4, uchar4>::add(uint4 &s, const uchar4 &v, const int c) { s.x += (int)v.x * c; s.y += (int)v.y * c; s.z += (int)v.z * c; s.w += (int)v.w * c; }
template<> __forceinline void CenterBlock<unsigned long long, ulonglong4, ushort4>::add(ulonglong4 &s, const ushort4 &v, const int c) { s.x += (long long)v.x * (long long)c; s.y += (long long)v.y * (long long)c; s.z += (long long)v.z * (long long)c; s.w += (long long)v.w * (long long)c; }

template<> __forceinline void CenterBlock<unsigned long long, uint, uchar>::convert(uchar &v, const uint &s) {
	uchar t = ((s + (weight >> 1)) / weight);
	changedCenter |= (t != v);
	v = t;
	squared += dot(t, t);
}

template<> __forceinline void CenterBlock<unsigned long long, unsigned long long, unsigned short>::convert(unsigned short &v, const unsigned long long &s) {
	unsigned short t = (unsigned short)((s + (weight >> 1)) / weight);
	changedCenter |= (t != v);
	v = t;
	squared += dot(t, t);
}

template<> __forceinline void CenterBlock<unsigned long long, uint4, uchar4>::convert(uchar4 &v, const uint4 &s) {
	uchar4 t = make_uchar4(((s.x + (weight >> 1)) / weight), ((s.y + (weight >> 1)) / weight), ((s.z + (weight >> 1)) / weight), ((s.w + (weight >> 1)) / weight));
	changedCenter |= !((t.x == v.x) && (t.y == v.y) && (t.z == v.z) && (t.w == v.w));
	v = t;
	squared += dot(t, t);
}

template<> __forceinline void CenterBlock<unsigned long long, ulonglong4, ushort4>::convert(ushort4 &v, const ulonglong4 &s) {
	ushort4 t = make_ushort4(
		(unsigned short)((s.x + (weight >> 1)) / weight), 
		(unsigned short)((s.y + (weight >> 1)) / weight), 
		(unsigned short)((s.z + (weight >> 1)) / weight), 
		(unsigned short)((s.w + (weight >> 1)) / weight));
	changedCenter |= !((t.x == v.x) && (t.y == v.y) && (t.z == v.z) && (t.w == v.w));
	v = t;
	squared += dot(t, t);
}

template<> __forceinline void CenterBlock<unsigned long long, uint, uchar>::assign(uchar &v, const uint &s) { v = s; }
template<> __forceinline void CenterBlock<unsigned long long, unsigned long long, unsigned short>::assign(unsigned short &v, const unsigned long long &s) { v = (unsigned short)s; }
template<> __forceinline void CenterBlock<unsigned long long, uint4, uchar4>::assign(uchar4 &v, const uint4 &s) { v = make_uchar4(s.x, s.y, s.z, s.w); }
template<> __forceinline void CenterBlock<unsigned long long, ulonglong4, ushort4>::assign(ushort4 &v, const ulonglong4 &s) { v = make_ushort4((unsigned short)s.x, (unsigned short)s.y, (unsigned short)s.z, (unsigned short)s.w); }

template <typename T>
class LessThan
{
public:
	LessThan() { }
	bool operator() (const T& lhs, const T&rhs) const
	{
		return (lhs.p < rhs.p);
	}
};

template <class A>
class IndexedPriority
{
public:
	A p;
	unsigned int idx;
	unsigned int count;
	IndexedPriority(unsigned int i, unsigned int c, A pri) { idx = i; p = pri; count = c; }
	IndexedPriority() {}
};

template <class VOL, class T> float calcPsnr(T &vol, VOL &sel, int v);

template <> float calcPsnr<CenterBlock<unsigned long long, uint, uchar>, unsigned char>(unsigned char &vol, CenterBlock<unsigned long long, uint, uchar> &sel, int v)
{
	float psnr = 0.0f;
	uchar val = sel.value[v];
	psnr = (float)(((int)vol - (int)val) * ((int)vol - (int)val));
	vol = (unsigned char)val;
	return psnr;
}

template <> float calcPsnr<CenterBlock<unsigned long long, unsigned long long, unsigned short>, unsigned short>(unsigned short &vol, CenterBlock<unsigned long long, unsigned long long, unsigned short> &sel, int v)
{
	float psnr = 0.0f;
	unsigned short val = sel.value[v];
	psnr = (float)(((int)vol - (int)val) * ((int)vol - (int)val));
	vol = val;
	return psnr;
}

template <> float calcPsnr<CenterBlock<unsigned long long, uint4, uchar4>, uchar4>(uchar4 &vol, CenterBlock<unsigned long long, uint4, uchar4> &sel, int v)
{
	float psnr = 0.0f;
	uchar4 val = sel.value[v];
	psnr  = (float)(((int)vol.x - (int)val.x) * ((int)vol.x - (int)val.x));
	psnr += (float)(((int)vol.y - (int)val.y) * ((int)vol.y - (int)val.y));
	psnr += (float)(((int)vol.z - (int)val.z) * ((int)vol.z - (int)val.z));
	psnr += (float)(((int)vol.w - (int)val.w) * ((int)vol.w - (int)val.w));
	vol = val;
	return psnr;
}

template <> float calcPsnr<CenterBlock<unsigned long long, ulonglong4, ushort4>, ushort4>(ushort4 &vol, CenterBlock<unsigned long long, ulonglong4, ushort4> &sel, int v)
{
	float psnr = 0.0f;
	ushort4 val = sel.value[v];
	psnr  = (float)(((int)vol.x - (int)val.x) * ((int)vol.x - (int)val.x));
	psnr += (float)(((int)vol.y - (int)val.y) * ((int)vol.y - (int)val.y));
	psnr += (float)(((int)vol.z - (int)val.z) * ((int)vol.z - (int)val.z));
	psnr += (float)(((int)vol.w - (int)val.w) * ((int)vol.w - (int)val.w));
	vol = val;
	return psnr;
}

template <class T, class A, class D, class O>
void quantizeVolume(T *vol, cudaExtent &volumeSize, int lossy, bool bruteForce)
{
	if (lossy == 0) return;
#ifdef WIN32
	LARGE_INTEGER start, end;
	QueryPerformanceCounter(&start);
#else
	std::chrono::high_resolution_clock::time_point start, end;
	start = std::chrono::high_resolution_clock::now();
#endif
	std::vector<VolumeBlock<A, O>> blocks;
	unsigned int unique_blocks = 0;
	{
		// create blocks
		std::vector<VolumeBlock<A, O>> *hash_blocks = new std::vector<VolumeBlock<A, O>>[65536];
		for (unsigned int z0 = 0; z0 < volumeSize.depth; z0 += 4)
		{
			for (unsigned int y0 = 0; y0 < volumeSize.height; y0 += 4)
			{
				for (unsigned int x0 = 0; x0 < volumeSize.width; x0 += 4)
				{
					VolumeBlock<A, O> tmp;
					for (unsigned int z1 = 0; z1 < 4; z1++)
					{
						unsigned int z = std::min((unsigned int)(volumeSize.depth - 1), z0 + z1);
						for (unsigned int y1 = 0; y1 < 4; y1++)
						{
							unsigned int y = std::min((unsigned int)(volumeSize.height - 1), y0 + y1);
							for (unsigned int x1 = 0; x1 < 4; x1++)
							{
								unsigned int x = std::min((unsigned int)(volumeSize.width - 1), x0 + x1);
								//tmp.value[x1 + y1 * 4 + z1 * 16] = convert(vol[x + (y + z * volumeSize.height) * volumeSize.width]);
								tmp.value[x1 + y1 * 4 + z1 * 16] = vol[x + (y + z * volumeSize.height) * volumeSize.width];
							}
						}
					}
					tmp.calcHash();
					tmp.count = 1;
					tmp.last = 0;
					bool unique = true;
					for (unsigned int i = 0; (unique) && (i < hash_blocks[tmp.hash].size()); i++)
					{
						if (tmp == hash_blocks[tmp.hash][i])
						{
							unique = false;
							hash_blocks[tmp.hash][i].count++;
						}
					}
					if (unique)
					{
						hash_blocks[tmp.hash].push_back(tmp);
						unique_blocks++;
					}
				}
			}
		}
		blocks.resize(unique_blocks);
		unsigned int i = 0;

		for (unsigned int list = 0; list < 65536; list++)
		{
			for (unsigned int idx = 0; idx < hash_blocks[list].size(); idx++)
			{
				blocks[i++] = hash_blocks[list][idx];
			}
		}
		delete[] hash_blocks;
	}

	unsigned int k = (unique_blocks + ((1 << lossy) - 1)) >> lossy;
	std::cout << unique_blocks << " unique blocks, quantizing to " << k << " blocks." << std::endl;

	std::vector<CenterBlock<A, D, O>> selected_blocks(k);
	if (bruteForce)
	{
		A global_min = A(0x7fffffffffffffffll);
		unsigned int min_idx = 0;

#pragma omp parallel for
		for (int i = 0; i < (int)unique_blocks; i++)
		{
			A p;
			p = blocks[i].dist();// * blocks[i].count;
			if (p <= global_min)
			{
#pragma omp critical
				{
					if ((p < global_min) || ((p == global_min) && (i < (int)min_idx)))
					{
						global_min = p;
						min_idx = i;
					}
				}
			}
		}
		selected_blocks[0] = blocks[min_idx];
		int next_idx = 0;
		A global_max = A(0);
		// update distances
		{
			CenterBlock<A, D, O> &center = selected_blocks[0];
#pragma omp parallel for
			for (int i = 0; i < (int)unique_blocks; i++)
			{
				VolumeBlock<A, O> &blk = blocks[i];
				A p = blk.lastDist = center.dist(blk);
				blk.last = 0;
				if (p >= global_max)
				{
#pragma omp critical
					{
						if ((p > global_max) || ((p == global_max) && (i < (int)next_idx)))
						{
							global_max = p;
							next_idx = i;
						}
					}
				}
			}
		}
		unsigned int done = 1;
		while (done < k)
		{
			selected_blocks[done] = blocks[next_idx];
			// update distances
			global_max = A(0);
			next_idx = 0;
			CenterBlock<A, D, O> &center = selected_blocks[done];
#pragma omp parallel for
			for (int i = 0; i < (int)unique_blocks; i++)
			{
				VolumeBlock<A, O> &blk = blocks[i];
				if (center.dist(blk, blk.lastDist, false))
				{
					blk.last = done;
				}
				A p = blk.lastDist;
				if ((done + 1 < k) && (p >= global_max))
				{
#pragma omp critical
					{
						if ((p > global_max) || ((p == global_max) && (i < (int)next_idx)))
						{
							global_max = p;
							next_idx = i;
						}
					}
				}
			}
			done++;
			if ((done * 80 / k) > ((done - 1) * 80 / k)) std::cout << ".";
		}
	}
	else
	{
		std::vector<IndexedPriority<A>> pri_container(unique_blocks);
		std::vector<unsigned int> selection(unique_blocks);
		A global_min = A(0x7fffffffffffffffll);
		unsigned int min_idx = 0;

#pragma omp parallel for
		for (int i = 0; i < (int)unique_blocks; i++)
		{
			A p;
			p = blocks[i].dist();// * blocks[i].count;
			pri_container[i].count = 0;
			pri_container[i].idx = i;
			pri_container[i].p = p;
			if (p <= global_min)
			{
#pragma omp critical
				{
					if ((p < global_min) || ((p == global_min) && (i < (int)min_idx)))
					{
						global_min = p;
						min_idx = i;
					}
				}
			}
		}

		std::swap(pri_container[min_idx], pri_container[unique_blocks - 1]);
		pri_container.pop_back();
		selection[0] = min_idx;

		std::priority_queue<IndexedPriority<A>, std::vector<IndexedPriority<A>>, LessThan<IndexedPriority<A>>> pri_queue(LessThan<IndexedPriority<A>>(), pri_container);

		unsigned int done = 1;
		unsigned int last_done = 0;
		while (done < k)
		{
			IndexedPriority<A> tmp = pri_queue.top();
			pri_queue.pop();

			A p = tmp.p;// / blocks[tmp.idx].count;
			int total = done - tmp.count;
#pragma omp parallel for
			for (int idx = 0; idx < total; idx++)
			{
				A pp;
				if (blocks[tmp.idx].dist(blocks[selection[tmp.count + idx]], p, pp))
				{
					if (pp < p)
					{
#pragma omp critical
						{
							if (pp < p)
							{
								p = pp;
							}
						}
					}
				}
			}
			tmp.p = p;// * blocks[tmp.idx].count;
			tmp.count = done;

			if (tmp.p >= pri_queue.top().p)
			{
				// still the furthest
				selection[done++] = tmp.idx;
				if ((done * 80 / k) > (last_done * 80 / k)) std::cout << ".";
				last_done = done;
			}
			else
			{
				// push back onto queue
				pri_queue.push(tmp);
			}
		}
		for (unsigned int i = 0; i < k; i++)
		{
			selected_blocks[i] = blocks[selection[i]];
		}
	}
	std::cout << std::endl;

	std::cout << "doing relaxation: " << std::endl;

	unsigned int *changed_blocks = new unsigned int[k];
	for (unsigned int i = 0; i < k; i++) changed_blocks[i] = i;
	unsigned int changed_blocks_size = k;

	unsigned int changed = changed_blocks_size;

	int run = 0;

	double psnr_tmp = 0.0;

	while ((changed > 0) && (run < 1000))
	{
		changed = 0;
		unsigned int switched = 0;
		double psnr_dlt = 0.0;
#pragma omp parallel for reduction(+: psnr_dlt, switched)
		for (int idx = 0; idx < (int)unique_blocks; idx++)
		{
			VolumeBlock<A, O> &blk = blocks[idx];
			CenterBlock<A, D, O> &center = selected_blocks[blk.last];
			unsigned int c = blk.last;
			A min_dist = A(0);

			if ((run == 0) && (bruteForce))
			{
				min_dist = blk.lastDist;
			}
			else
			{
				if (center.changedCenter)
				{
					min_dist = center.dist(blk);
				}
				else
					min_dist = blk.lastDist;

				if ((center.changedCenter) && (min_dist > blk.lastDist))
				{
					// increased distance so need to check all
					for (unsigned int i = 0; i < k; i++)
					{
						if ((min_dist == 0) && (i > c)) break;
						if (i != blk.last)
						{
							CenterBlock<A, D, O> &comp = selected_blocks[i];
							if (comp.dist(blk, min_dist, (i < c)))
							{
								c = i;
							}
						}
					}
				}
				else
				{
					// distance decreased or unchanged, only check modified blocks
					for (unsigned int ii = 0; ii < changed_blocks_size; ii++)
					{
						unsigned int i = changed_blocks[ii];
						if ((min_dist == 0) && (i > c)) break;
						if (i != blk.last)
						{
							CenterBlock<A, D, O> &comp = selected_blocks[i];
							if (comp.dist(blk, min_dist, (i < c)))
							{
								c = i;
							}
						}
					}
				}
			}
			CenterBlock<A, D, O> &updated = selected_blocks[c];

			if ((blk.last != c) || (run == 0))
			{
				switched++;
				if (run > 0)
				{
					center.sub(blk);
				}
				updated.add(blk);
				center.changedMember = true;
				updated.changedMember = true;
				blk.last = c;
			}
			{
				double add = 0.0;
				//if (run > 0) add -= (double)blk.lastDist * (double)blk.count;
				add += (double)min_dist * (double)blk.count;
				psnr_dlt += add;
			}
			blk.lastDist = min_dist;
		}
		psnr_tmp = psnr_dlt;
		changed_blocks_size = 0;
#pragma omp parallel for
		for (int i = 0; i < (int)k; i++)
		{
			selected_blocks[i].update();
		}

		for (int i = 0; i < (int)k; i++)
		{
			if (selected_blocks[i].changedCenter)
			{
				changed_blocks[changed_blocks_size++] = i;
			}
		}
		// calculate PSNR
		{
			float psnr = (float)psnr_tmp;
			float a = quantizeMaximum(vol);
			float b = 0.0f;
			psnr /= (float)(volumeSize.width * volumeSize.height * volumeSize.depth);
			psnr = (a - b) * (a - b) / psnr;
			psnr = 10.0f * logf(psnr) / logf(10.0f);
			std::cout << "  PSNR: " << psnr << "db" << std::endl;
		}
		changed = changed_blocks_size;
		run++;
		std::cout << "  iteration " << run << ": " << changed << " blocks changed (" << switched << " switched owner)." << std::endl;
	}

	// recreate hashed lookup table instead of brute force search again
	std::vector<VolumeBlock<A, O>> *hash_blocks = new std::vector<VolumeBlock<A, O>>[65536];
	for (unsigned int i = 0; i < blocks.size(); i++)
	{
		hash_blocks[blocks[i].hash].push_back(blocks[i]);
	}

	float psnr = 0.0f;
	for (unsigned int z0 = 0; z0 < volumeSize.depth; z0 += 4)
	{
		for (unsigned int y0 = 0; y0 < volumeSize.height; y0 += 4)
		{
			for (unsigned int x0 = 0; x0 < volumeSize.width; x0 += 4)
			{
				VolumeBlock<A, O> tmp;
				for (unsigned int z1 = 0; z1 < 4; z1++)
				{
					unsigned int z = std::min((unsigned int)(volumeSize.depth - 1), z0 + z1);
					for (unsigned int y1 = 0; y1 < 4; y1++)
					{
						unsigned int y = std::min((unsigned int)(volumeSize.height - 1), y0 + y1);
						for (unsigned int x1 = 0; x1 < 4; x1++)
						{
							unsigned int x = std::min((unsigned int)(volumeSize.width - 1), x0 + x1);
							//tmp.value[x1 + y1 * 4 + z1 * 16] = convert(vol[x + (y + z * volumeSize.height) * volumeSize.width]);
							tmp.value[x1 + y1 * 4 + z1 * 16] = vol[x + (y + z * volumeSize.height) * volumeSize.width];
						}
					}
				}
				tmp.calcHash();
				unsigned int c = 0;
				for (unsigned int i = 0; i < hash_blocks[tmp.hash].size(); i++)
				{
					if (tmp == hash_blocks[tmp.hash][i]) c = hash_blocks[tmp.hash][i].last;
				}
				// strictly speaking this doesn't work since the data outside the volume might not match the data at the border.
				for (unsigned int z1 = 0; (z1 < 4) && (z0 + z1 < volumeSize.depth); z1++)
				{
					unsigned int z = z0 + z1;
					for (unsigned int y1 = 0; (y1 < 4) && (y0 + y1 < volumeSize.height); y1++)
					{
						unsigned int y = y0 + y1;
						for (unsigned int x1 = 0; (x1 < 4) && (x0 + x1 < volumeSize.width); x1++)
						{
							unsigned int x = x0 + x1;
							{
								int v = x1 + y1 * 4 + z1 * 16;
								psnr += calcPsnr<CenterBlock<A, D, O>, T>(vol[x + (y + z * volumeSize.height) * volumeSize.width], selected_blocks[c], v);
							}
						}
					}
				}
			}
		}
	}

	delete[] hash_blocks;

	float a = quantizeMaximum(vol);
	float b = 0.0f;
	psnr /= (float)(volumeSize.width * volumeSize.height * volumeSize.depth);
	psnr = (a - b) * (a - b) / psnr;
	psnr = 10.0f * logf(psnr) / logf(10.0f);
	std::cout << "PSNR: " << psnr << "db" << std::endl;
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
	std::cout << time << "ns = " << time / 1000000.0 << "s = " << time / 60000000.0 << "min" << std::endl;
}

template <class T>
T sampleVolume(T *vol, float3 &w, int3 &smp0, int3 &smp1, cudaExtent &volumeSize);

template <>
unsigned char sampleVolume<unsigned char>(unsigned char *vol, float3 &w, int3 &smp0, int3 &smp1, cudaExtent &volumeSize)
{
	float s = (1.0f - w.z) * (1.0f - w.y) * (1.0f - w.x) * vol[smp0.x + volumeSize.width * (smp0.y + volumeSize.height * smp0.z)]
			+ (1.0f - w.z) * (1.0f - w.y) *         w.x  * vol[smp1.x + volumeSize.width * (smp0.y + volumeSize.height * smp0.z)]
			+ (1.0f - w.z) *         w.y  * (1.0f - w.x) * vol[smp0.x + volumeSize.width * (smp1.y + volumeSize.height * smp0.z)]
			+ (1.0f - w.z) *         w.y  *         w.x  * vol[smp1.x + volumeSize.width * (smp1.y + volumeSize.height * smp0.z)]
			+         w.z  * (1.0f - w.y) * (1.0f - w.x) * vol[smp0.x + volumeSize.width * (smp0.y + volumeSize.height * smp1.z)]
			+         w.z  * (1.0f - w.y) *         w.x  * vol[smp1.x + volumeSize.width * (smp0.y + volumeSize.height * smp1.z)]
			+         w.z  *         w.y  * (1.0f - w.x) * vol[smp0.x + volumeSize.width * (smp1.y + volumeSize.height * smp1.z)]
			+         w.z  *         w.y  *         w.x  * vol[smp1.x + volumeSize.width * (smp1.y + volumeSize.height * smp1.z)];
	return (unsigned char)floor(s + 0.5f);
}

template <>
unsigned short sampleVolume<unsigned short>(unsigned short *vol, float3 &w, int3 &smp0, int3 &smp1, cudaExtent &volumeSize)
{
	float s = (1.0f - w.z) * (1.0f - w.y) * (1.0f - w.x) * vol[smp0.x + volumeSize.width * (smp0.y + volumeSize.height * smp0.z)]
			+ (1.0f - w.z) * (1.0f - w.y) *         w.x  * vol[smp1.x + volumeSize.width * (smp0.y + volumeSize.height * smp0.z)]
			+ (1.0f - w.z) *         w.y  * (1.0f - w.x) * vol[smp0.x + volumeSize.width * (smp1.y + volumeSize.height * smp0.z)]
			+ (1.0f - w.z) *         w.y  *         w.x  * vol[smp1.x + volumeSize.width * (smp1.y + volumeSize.height * smp0.z)]
			+         w.z  * (1.0f - w.y) * (1.0f - w.x) * vol[smp0.x + volumeSize.width * (smp0.y + volumeSize.height * smp1.z)]
			+         w.z  * (1.0f - w.y) *         w.x  * vol[smp1.x + volumeSize.width * (smp0.y + volumeSize.height * smp1.z)]
			+         w.z  *         w.y  * (1.0f - w.x) * vol[smp0.x + volumeSize.width * (smp1.y + volumeSize.height * smp1.z)]
			+         w.z  *         w.y  *         w.x  * vol[smp1.x + volumeSize.width * (smp1.y + volumeSize.height * smp1.z)];
	return (unsigned short)floor(s + 0.5f);
}

template <>
uchar4 sampleVolume<uchar4>(uchar4 *vol, float3 &w, int3 &smp0, int3 &smp1, cudaExtent &volumeSize)
{
	float4 s;
	s.x = (1.0f - w.z) * (1.0f - w.y) * (1.0f - w.x) * vol[smp0.x + volumeSize.width * (smp0.y + volumeSize.height * smp0.z)].x
		+ (1.0f - w.z) * (1.0f - w.y) *         w.x  * vol[smp1.x + volumeSize.width * (smp0.y + volumeSize.height * smp0.z)].x
		+ (1.0f - w.z) *         w.y  * (1.0f - w.x) * vol[smp0.x + volumeSize.width * (smp1.y + volumeSize.height * smp0.z)].x
		+ (1.0f - w.z) *         w.y  *         w.x  * vol[smp1.x + volumeSize.width * (smp1.y + volumeSize.height * smp0.z)].x
		+         w.z  * (1.0f - w.y) * (1.0f - w.x) * vol[smp0.x + volumeSize.width * (smp0.y + volumeSize.height * smp1.z)].x
		+         w.z  * (1.0f - w.y) *         w.x  * vol[smp1.x + volumeSize.width * (smp0.y + volumeSize.height * smp1.z)].x
		+         w.z  *         w.y  * (1.0f - w.x) * vol[smp0.x + volumeSize.width * (smp1.y + volumeSize.height * smp1.z)].x
		+         w.z  *         w.y  *         w.x  * vol[smp1.x + volumeSize.width * (smp1.y + volumeSize.height * smp1.z)].x;
	s.y = (1.0f - w.z) * (1.0f - w.y) * (1.0f - w.x) * vol[smp0.x + volumeSize.width * (smp0.y + volumeSize.height * smp0.z)].y
		+ (1.0f - w.z) * (1.0f - w.y) *         w.x  * vol[smp1.x + volumeSize.width * (smp0.y + volumeSize.height * smp0.z)].y
		+ (1.0f - w.z) *         w.y  * (1.0f - w.x) * vol[smp0.x + volumeSize.width * (smp1.y + volumeSize.height * smp0.z)].y
		+ (1.0f - w.z) *         w.y  *         w.x  * vol[smp1.x + volumeSize.width * (smp1.y + volumeSize.height * smp0.z)].y
		+         w.z  * (1.0f - w.y) * (1.0f - w.x) * vol[smp0.x + volumeSize.width * (smp0.y + volumeSize.height * smp1.z)].y
		+         w.z  * (1.0f - w.y) *         w.x  * vol[smp1.x + volumeSize.width * (smp0.y + volumeSize.height * smp1.z)].y
		+         w.z  *         w.y  * (1.0f - w.x) * vol[smp0.x + volumeSize.width * (smp1.y + volumeSize.height * smp1.z)].y
		+         w.z  *         w.y  *         w.x  * vol[smp1.x + volumeSize.width * (smp1.y + volumeSize.height * smp1.z)].y;
	s.z = (1.0f - w.z) * (1.0f - w.y) * (1.0f - w.x) * vol[smp0.x + volumeSize.width * (smp0.y + volumeSize.height * smp0.z)].z
		+ (1.0f - w.z) * (1.0f - w.y) *         w.x  * vol[smp1.x + volumeSize.width * (smp0.y + volumeSize.height * smp0.z)].z
		+ (1.0f - w.z) *         w.y  * (1.0f - w.x) * vol[smp0.x + volumeSize.width * (smp1.y + volumeSize.height * smp0.z)].z
		+ (1.0f - w.z) *         w.y  *         w.x  * vol[smp1.x + volumeSize.width * (smp1.y + volumeSize.height * smp0.z)].z
		+         w.z  * (1.0f - w.y) * (1.0f - w.x) * vol[smp0.x + volumeSize.width * (smp0.y + volumeSize.height * smp1.z)].z
		+         w.z  * (1.0f - w.y) *         w.x  * vol[smp1.x + volumeSize.width * (smp0.y + volumeSize.height * smp1.z)].z
		+         w.z  *         w.y  * (1.0f - w.x) * vol[smp0.x + volumeSize.width * (smp1.y + volumeSize.height * smp1.z)].z
		+         w.z  *         w.y  *         w.x  * vol[smp1.x + volumeSize.width * (smp1.y + volumeSize.height * smp1.z)].z;
	s.w = (1.0f - w.z) * (1.0f - w.y) * (1.0f - w.x) * vol[smp0.x + volumeSize.width * (smp0.y + volumeSize.height * smp0.z)].w
		+ (1.0f - w.z) * (1.0f - w.y) *         w.x  * vol[smp1.x + volumeSize.width * (smp0.y + volumeSize.height * smp0.z)].w
		+ (1.0f - w.z) *         w.y  * (1.0f - w.x) * vol[smp0.x + volumeSize.width * (smp1.y + volumeSize.height * smp0.z)].w
		+ (1.0f - w.z) *         w.y  *         w.x  * vol[smp1.x + volumeSize.width * (smp1.y + volumeSize.height * smp0.z)].w
		+         w.z  * (1.0f - w.y) * (1.0f - w.x) * vol[smp0.x + volumeSize.width * (smp0.y + volumeSize.height * smp1.z)].w
		+         w.z  * (1.0f - w.y) *         w.x  * vol[smp1.x + volumeSize.width * (smp0.y + volumeSize.height * smp1.z)].w
		+         w.z  *         w.y  * (1.0f - w.x) * vol[smp0.x + volumeSize.width * (smp1.y + volumeSize.height * smp1.z)].w
		+         w.z  *         w.y  *         w.x  * vol[smp1.x + volumeSize.width * (smp1.y + volumeSize.height * smp1.z)].w;
	return make_uchar4((unsigned char)floor(s.x + 0.5f), (unsigned char)floor(s.y + 0.5f), (unsigned char)floor(s.z + 0.5f), (unsigned char)floor(s.w + 0.5f));
}

template <>
ushort4 sampleVolume<ushort4>(ushort4 *vol, float3 &w, int3 &smp0, int3 &smp1, cudaExtent &volumeSize)
{
	float4 s;
	s.x = (1.0f - w.z) * (1.0f - w.y) * (1.0f - w.x) * vol[smp0.x + volumeSize.width * (smp0.y + volumeSize.height * smp0.z)].x
		+ (1.0f - w.z) * (1.0f - w.y) *         w.x  * vol[smp1.x + volumeSize.width * (smp0.y + volumeSize.height * smp0.z)].x
		+ (1.0f - w.z) *         w.y  * (1.0f - w.x) * vol[smp0.x + volumeSize.width * (smp1.y + volumeSize.height * smp0.z)].x
		+ (1.0f - w.z) *         w.y  *         w.x  * vol[smp1.x + volumeSize.width * (smp1.y + volumeSize.height * smp0.z)].x
		+         w.z  * (1.0f - w.y) * (1.0f - w.x) * vol[smp0.x + volumeSize.width * (smp0.y + volumeSize.height * smp1.z)].x
		+         w.z  * (1.0f - w.y) *         w.x  * vol[smp1.x + volumeSize.width * (smp0.y + volumeSize.height * smp1.z)].x
		+         w.z  *         w.y  * (1.0f - w.x) * vol[smp0.x + volumeSize.width * (smp1.y + volumeSize.height * smp1.z)].x
		+         w.z  *         w.y  *         w.x  * vol[smp1.x + volumeSize.width * (smp1.y + volumeSize.height * smp1.z)].x;
	s.y = (1.0f - w.z) * (1.0f - w.y) * (1.0f - w.x) * vol[smp0.x + volumeSize.width * (smp0.y + volumeSize.height * smp0.z)].y
		+ (1.0f - w.z) * (1.0f - w.y) *         w.x  * vol[smp1.x + volumeSize.width * (smp0.y + volumeSize.height * smp0.z)].y
		+ (1.0f - w.z) *         w.y  * (1.0f - w.x) * vol[smp0.x + volumeSize.width * (smp1.y + volumeSize.height * smp0.z)].y
		+ (1.0f - w.z) *         w.y  *         w.x  * vol[smp1.x + volumeSize.width * (smp1.y + volumeSize.height * smp0.z)].y
		+         w.z  * (1.0f - w.y) * (1.0f - w.x) * vol[smp0.x + volumeSize.width * (smp0.y + volumeSize.height * smp1.z)].y
		+         w.z  * (1.0f - w.y) *         w.x  * vol[smp1.x + volumeSize.width * (smp0.y + volumeSize.height * smp1.z)].y
		+         w.z  *         w.y  * (1.0f - w.x) * vol[smp0.x + volumeSize.width * (smp1.y + volumeSize.height * smp1.z)].y
		+         w.z  *         w.y  *         w.x  * vol[smp1.x + volumeSize.width * (smp1.y + volumeSize.height * smp1.z)].y;
	s.z = (1.0f - w.z) * (1.0f - w.y) * (1.0f - w.x) * vol[smp0.x + volumeSize.width * (smp0.y + volumeSize.height * smp0.z)].z
		+ (1.0f - w.z) * (1.0f - w.y) *         w.x  * vol[smp1.x + volumeSize.width * (smp0.y + volumeSize.height * smp0.z)].z
		+ (1.0f - w.z) *         w.y  * (1.0f - w.x) * vol[smp0.x + volumeSize.width * (smp1.y + volumeSize.height * smp0.z)].z
		+ (1.0f - w.z) *         w.y  *         w.x  * vol[smp1.x + volumeSize.width * (smp1.y + volumeSize.height * smp0.z)].z
		+         w.z  * (1.0f - w.y) * (1.0f - w.x) * vol[smp0.x + volumeSize.width * (smp0.y + volumeSize.height * smp1.z)].z
		+         w.z  * (1.0f - w.y) *         w.x  * vol[smp1.x + volumeSize.width * (smp0.y + volumeSize.height * smp1.z)].z
		+         w.z  *         w.y  * (1.0f - w.x) * vol[smp0.x + volumeSize.width * (smp1.y + volumeSize.height * smp1.z)].z
		+         w.z  *         w.y  *         w.x  * vol[smp1.x + volumeSize.width * (smp1.y + volumeSize.height * smp1.z)].z;
	s.w = (1.0f - w.z) * (1.0f - w.y) * (1.0f - w.x) * vol[smp0.x + volumeSize.width * (smp0.y + volumeSize.height * smp0.z)].w
		+ (1.0f - w.z) * (1.0f - w.y) *         w.x  * vol[smp1.x + volumeSize.width * (smp0.y + volumeSize.height * smp0.z)].w
		+ (1.0f - w.z) *         w.y  * (1.0f - w.x) * vol[smp0.x + volumeSize.width * (smp1.y + volumeSize.height * smp0.z)].w
		+ (1.0f - w.z) *         w.y  *         w.x  * vol[smp1.x + volumeSize.width * (smp1.y + volumeSize.height * smp0.z)].w
		+         w.z  * (1.0f - w.y) * (1.0f - w.x) * vol[smp0.x + volumeSize.width * (smp0.y + volumeSize.height * smp1.z)].w
		+         w.z  * (1.0f - w.y) *         w.x  * vol[smp1.x + volumeSize.width * (smp0.y + volumeSize.height * smp1.z)].w
		+         w.z  *         w.y  * (1.0f - w.x) * vol[smp0.x + volumeSize.width * (smp1.y + volumeSize.height * smp1.z)].w
		+         w.z  *         w.y  *         w.x  * vol[smp1.x + volumeSize.width * (smp1.y + volumeSize.height * smp1.z)].w;
	return make_ushort4((unsigned short)floor(s.x + 0.5f), (unsigned short)floor(s.y + 0.5f), (unsigned short)floor(s.z + 0.5f), (unsigned short)floor(s.w + 0.5f));
}

template <class T>
void resampleVolume(T *vol, T *out, cudaExtent &volumeSize, cudaExtent &resampleSize)
{
	float3 pos;
	float3 w;
	int3 smp0, smp1;
	for (size_t z = 0; z < resampleSize.depth; z++)
	{
		pos.z = (float)(z * (volumeSize.depth - 1)) / (float)(resampleSize.depth - 1);
		smp0.z = (int)floor(pos.z);
		smp1.z = std::min(smp0.z + 1, (int)volumeSize.depth - 1);
		w.z = pos.z - smp0.z;
		for (size_t y = 0; y < resampleSize.height; y++)
		{
			pos.y = (float)(y * (volumeSize.height - 1)) / (float)(resampleSize.height - 1);
			smp0.y = (int)floor(pos.y);
			smp1.y = std::min(smp0.y + 1, (int)volumeSize.height - 1);
			w.y = pos.y - smp0.y;
			for (size_t x = 0; x < resampleSize.width; x++)
			{
				pos.x = (float)(x * (volumeSize.width - 1)) / (float)(resampleSize.width - 1);
				smp0.x = (int)floor(pos.x);
				smp1.x = std::min(smp0.x + 1, (int)volumeSize.width - 1);
				w.x = pos.x - smp0.x;
				out[x + resampleSize.width * (y + resampleSize.height * z)] = sampleVolume<T>(vol, w, smp0, smp1, volumeSize);
			}
		}
	}
}

//#define QUICK_RESAMPLE

template <class T>
void expandVolume(T *vol, T *out, cudaExtent &volumeSize, cudaExtent &resampleSize)
{
	std::cout << "expanding volume" << std::endl;
#ifdef WIN32
	LARGE_INTEGER start, end;
	QueryPerformanceCounter(&start);
#else
	std::chrono::high_resolution_clock::time_point start, end;
	start = std::chrono::high_resolution_clock::now();
#endif
#ifdef QUICK_RESAMPLE
	for (size_t z = 0; z < resampleSize.depth; z++)
	{
		size_t z0 = std::min(z, volumeSize.depth - 1);
		for (size_t y = 0; y < resampleSize.height; y++)
		{
			size_t y0 = std::min(y, volumeSize.height - 1);
			for (size_t x = 0; x < resampleSize.width; x++)
			{
				size_t x0 = std::min(x, volumeSize.width - 1);
				out[x + resampleSize.width * (y + resampleSize.height * z)] = vol[x0 + volumeSize.width * (y0 + volumeSize.height * z0)];
			}
		}
	}
#else
	std::vector<std::vector<std::pair<std::pair<size_t, size_t>, size_t>>> loc(65536);
	for (size_t z = 0; z < resampleSize.depth; z += 4)
	{
		for (size_t y = 0; y < resampleSize.height; y += 4)
		{
			for (size_t x = 0; x < resampleSize.width; x += 4)
			{
				if ((z + 3 < volumeSize.depth) && (y + 3 < volumeSize.height) && (x + 3 < volumeSize.width))
				{
					std::pair<std::pair<size_t, size_t>, size_t> l;
					l.first.first = x;
					l.first.second = y;
					l.second = z;
					loc[vol[x + volumeSize.width * (y + volumeSize.height * z)]].push_back(l);
				}
			}
		}
	}
#pragma omp parallel for schedule(dynamic)
	for (long long i = 0; i < (long long)((resampleSize.width * resampleSize.height * resampleSize.depth) >> 6); i++)
//	for (long long zz = 0; zz < (long long)resampleSize.depth; zz += 4)
	{
		//size_t z = (size_t)zz;
		size_t z = (size_t)((i / ((resampleSize.width * resampleSize.height) >> 4)) << 2);
//#pragma omp parallel for
//		for (long long yy = 0; yy < (long long)resampleSize.height; yy += 4)
		{
			//size_t y = (size_t)yy;
			size_t y = (size_t)(((i / (resampleSize.width >> 2)) % (resampleSize.height >> 2)) << 2);
//#pragma omp parallel for
//			for (long long xx = 0; xx < (long long)resampleSize.width; xx+=4)
			{
				//size_t x = (size_t)xx;
				size_t x = (size_t)((i % (resampleSize.width >> 2)) << 2);
				if ((z + 3 < volumeSize.depth) && (y + 3 < volumeSize.height) && (x + 3 < volumeSize.width))
				{
					for (size_t z0 = z; z0 < z + 4; z0++)
					{
						for (size_t y0 = y; y0 < y + 4; y0++)
						{
							for (size_t x0 = x; x0 < x + 4; x0++)
							{
								out[x0 + resampleSize.width * (y0 + resampleSize.height * z0)] = vol[x0 + volumeSize.width * (y0 + volumeSize.height * z0)];
							}
						}
					}
				}
				else
				{
					bool found = false;
					size_t xa, ya, za;

					T l = vol[x + volumeSize.width * (y + volumeSize.height * z)];

					//for (za = 0; (!found) && (za < resampleSize.depth); za += 4)
					for (size_t ii = 0; (!found) && (ii < loc[l].size()); ii++)
					{
						xa = loc[l][ii].first.first;
						ya = loc[l][ii].first.second;
						za = loc[l][ii].second;
						//if (za != z) for (ya = 0; (!found) && (ya < resampleSize.height); ya += 4)
						{
							//if (ya != y) for (xa = 0; (!found) && (xa < resampleSize.width); xa += 4)
							{
								if (xa != x) if ((za + 3 < volumeSize.depth) && (ya + 3 < volumeSize.height) && (xa + 3 < volumeSize.width))
								{
									found = true;
									for (size_t z0 = z; (z0 < z + 4) && (z0 < volumeSize.depth); z0++)
									{
										size_t z1 = z0 - z + za;
										for (size_t y0 = y; (y0 < y + 4) && (y0 < volumeSize.height); y0++)
										{
											size_t y1 = y0 - y + ya;
											for (size_t x0 = x; (x0 < x + 4) && (x0 < volumeSize.width); x0++)
											{
												size_t x1 = x0 - x + xa;
												if (vol[x0 + volumeSize.width * (y0 + volumeSize.height * z0)] != vol[x1 + volumeSize.width * (y1 + volumeSize.height * z1)])
													found = false;
											}
										}
									}
								}
							}
						}
					}

					if (found)
					{
						for (size_t z1 = z; z1 < z + 4; z1++)
						{
							size_t z0 = z1 - z + za;
							for (size_t y1 = y; y1 < y + 4; y1++)
							{
								size_t y0 = y1 - y + ya;
								for (size_t x1 = x; x1 < x + 4; x1++)
								{
									size_t x0 = x1 - x + xa;
									out[x1 + resampleSize.width * (y1 + resampleSize.height * z1)] = vol[x0 + volumeSize.width * (y0 + volumeSize.height * z0)];
								}
							}
						}
					}
					else
					{
						for (size_t z1 = z; z1 < z + 4; z1++)
						{
							size_t z0 = std::min(volumeSize.depth - 1, z1);
							for (size_t y1 = y; y1 < y + 4; y1++)
							{
								size_t y0 = std::min(volumeSize.height - 1, y1);
								for (size_t x1 = x; x1 < x + 4; x1++)
								{
									size_t x0 = std::min(volumeSize.width - 1, x1);
									out[x1 + resampleSize.width * (y1 + resampleSize.height * z1)] = vol[x0 + volumeSize.width * (y0 + volumeSize.height * z0)];
								}
							}
						}
					}
				}
			}
		}
	}
#endif
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
	std::cout << time << "ns = " << time / 1000000.0 << "s = " << time / 60000000.0 << "min" << std::endl;
}

void clipzero(unsigned char *vol, size_t size)
{
	for (size_t idx = 0; idx < size; idx++)
	{
		if (vol[idx + 3 * size] == 0)
			vol[idx] = vol[idx + size] = vol[idx + 2 * size] = vol[idx + 3 * size] = 0;
	}
}

void clipzero(unsigned short *vol, size_t size)
{
	for (size_t idx = 0; idx < size; idx++)
	{
		if (vol[idx + 3 * size] == 0)
			vol[idx] = vol[idx + size] = vol[idx + 2 * size] = vol[idx + 3 * size] = 0;
	}
}

void clipzero(uchar4 *vol, size_t size)
{
	for (size_t idx = 0; idx < size; idx++)
	{
		if (vol[idx].w == 0)
			vol[idx].x = vol[idx].y = vol[idx].z = vol[idx].w = 0;
	}
}

void setVol(unsigned char *vol_data, size_t off, size_t size, int val)
{
	vol_data[off] = val;
}

void setVol(unsigned short *vol_data, size_t off, size_t size, int val)
{
	vol_data[off] = val;
}

#ifdef LIBPNG_SUPPORT
unsigned int getPngElementSize(char *filename, int start)
{
	PngImage image;
	char f_in[1024];
	sprintf(f_in, filename, start);
	image.ReadImage(f_in);
	return image.GetBitDepth() >> 3;
}

unsigned int getPngComponents(char *filename, int start)
{
	PngImage image;
	char f_in[1024];
	sprintf(f_in, filename, start);
	image.ReadImage(f_in);
	return image.GetComponents();
}

// read a stack of png files (filename still has wildcards in it, start and end define the start end ending file names)
template <class T>
T* loadPngFiles(char *filename, cudaExtent &volumeSize, float3 &scale, int start, int end, int clip_x0, int clip_x1, int clip_y0, int clip_y1, float scale_png, bool clip_zero)
{
	int depth = end + 1 - start;
	T *vol_data = 0;

	PngImage image;
	char f_in[1024];
	sprintf(f_in, filename, start);
	image.ReadImage(f_in);
	volumeSize.width = (size_t) floor((image.GetWidth() - clip_x0 - clip_x1) / scale_png);
	volumeSize.height = (size_t)floor((image.GetHeight() - clip_y0 - clip_y1) / scale_png);
	// stack multiple components in z-axis
	volumeSize.depth = image.GetComponents() * depth;// / sizeof(T);

	size_t vol_size = volumeSize.width * volumeSize.height * volumeSize.depth * sizeof(T);

	vol_data = (T *)malloc(vol_size);

#pragma omp parallel for num_threads(12) private(f_in, image)
	for (int idx = start; idx <= end; idx++)
	{
		// we already loaded the first image
		//		if (idx != start)
		{
			sprintf(f_in, filename, idx);
			image.ReadImage(f_in);
		}
		for (size_t c = 0; c < image.GetComponents(); c++)
		{
			size_t z = c * depth + idx - start;
			for (int y = 0; y < (int)volumeSize.height; y++)
			{
				for (size_t x = 0; x < volumeSize.width; x++)
				{
					size_t off = x + (y + z * volumeSize.height) * volumeSize.width;
					unsigned int total = 0;
					unsigned int count = 0;
					for (int y0 = (int)floor(y * scale_png); y0 < floor((y + 1) * scale_png); y0++)
						for (int x0 = (int)floor(x * scale_png); x0 < floor((x + 1) * scale_png); x0++)
						{
							total += image.VGetValue((unsigned int)(x0 + clip_x0), (unsigned int)(y0 + clip_y0), (unsigned int)c);
							count++;
						}
					setVol(vol_data, off, volumeSize.width * volumeSize.height * volumeSize.depth * 4 / sizeof(T), (total + (count >> 1)) / count);
				}
			}
		}
	}
	if (clip_zero) clipzero(vol_data, volumeSize.width * volumeSize.height * volumeSize.depth * sizeof(T) / 4);

	return vol_data;
}
#endif

template <class T>
T *calculateGradients(T *vol, cudaExtent &volumeSize, unsigned int &components)
{
	if (components > 1) return vol;

	T *tmp = (T *)malloc(volumeSize.width * volumeSize.height * volumeSize.depth * 4 * sizeof(T));

	int xs[3], ys[3], zs[3];
	float off;
	for (size_t i = 0; i < volumeSize.width * volumeSize.height * volumeSize.depth; i++) off = std::max(off, (float) vol[i]);
	off = expf(logf(2.0f) * ceilf(logf(off - 1.0f) / logf(2.0f))) - 1.0f;
	off *= 0.5f;

	for (size_t z = 0; z < volumeSize.depth; z++)
	{
		zs[0] = std::max(0, (int)z - 1);
		zs[1] = (int) z;
		zs[2] = std::min((int)volumeSize.depth - 1, (int)z + 1);
		for (size_t y = 0; y < volumeSize.height; y++)
		{
			ys[0] = std::max(0, (int)y - 1);
			ys[1] = (int) y;
			ys[2] = std::min((int)volumeSize.height - 1, (int)y + 1);
#pragma omp parallel for private(xs)
			for (int x = 0; x < (int) volumeSize.width; x++)
			{
				xs[0] = std::max(0, (int)x - 1);
				xs[1] = (int) x;
				xs[2] = std::min((int)volumeSize.width - 1, (int)x + 1);
				float dx, dy, dz;
				float m[27];
				for (int i = 0; i < 27; i++)
					m[i] = (float)vol[xs[i % 3] + (ys[(i / 3) % 3] + zs[i / 9] * volumeSize.height) * volumeSize.width];
				dx =        m[ 2] + 2.0f * m[ 5] +        m[ 8] + 
					 2.0f * m[11] + 4.0f * m[14] + 2.0f * m[17] + 
					        m[20] + 2.0f * m[23] +        m[26] -
							m[ 0] - 2.0f * m[ 3] -        m[ 6] -
					 2.0f * m[ 9] - 4.0f * m[12] - 2.0f * m[15] -
							m[18] - 2.0f * m[21] -        m[24];
				dy =        m[ 6] + 2.0f * m[ 7] +        m[ 8] + 
					 2.0f * m[15] + 4.0f * m[16] + 2.0f * m[17] + 
					        m[24] + 2.0f * m[25] +        m[26] -
							m[ 0] - 2.0f * m[ 1] -        m[ 2] -
					 2.0f * m[ 9] - 4.0f * m[10] - 2.0f * m[11] -
							m[18] - 2.0f * m[19] -        m[20];
				dz =        m[18] + 2.0f * m[19] +        m[20] + 
					 2.0f * m[21] + 4.0f * m[22] + 2.0f * m[23] + 
					        m[24] + 2.0f * m[25] +        m[26] -
							m[ 0] - 2.0f * m[ 1] -        m[ 2] -
					 2.0f * m[ 3] - 4.0f * m[ 4] - 2.0f * m[ 5] -
							m[ 6] - 2.0f * m[ 7] -        m[ 8];
				dx = dx / 32.0f + off;
				dy = dy / 32.0f + off;
				dz = dz / 32.0f + off;
				tmp[x + (y +  z                         * volumeSize.height) * volumeSize.width] = (T) floor(dx + 0.5f);
				tmp[x + (y + (z +     volumeSize.depth) * volumeSize.height) * volumeSize.width] = (T) floor(dy + 0.5f);
				tmp[x + (y + (z + 2 * volumeSize.depth) * volumeSize.height) * volumeSize.width] = (T) floor(dz + 0.5f);
				tmp[x + (y + (z + 3 * volumeSize.depth) * volumeSize.height) * volumeSize.width] = vol[x + (y + z * volumeSize.height) * volumeSize.width];
			}
		}
	}
	std::swap(vol, tmp);
	free(tmp);
	components *= 4;
	return vol;
}

unsigned char getMax(unsigned char &x) { return x; }
unsigned short getMax(unsigned short &x) { return x; }
unsigned char getMax(uchar4 &x) { return std::max(std::max(x.x, x.y), std::max(x.z, x.w)); }
unsigned short getMax(ushort4 &x) { return std::max(std::max(x.x, x.y), std::max(x.z, x.w)); }

void doCount(size_t *count, unsigned char *v, size_t i) { count[v[i]]++; }
void doCount(size_t *count, unsigned short *v, size_t i) { count[v[i]]++; }
void doCount(size_t *count, uchar4 *v, size_t i) { count[v[i].x]++; count[v[i].y]++; count[v[i].z]++; count[v[i].w]++; }
void doCount(size_t *count, ushort4 *v, size_t i) { count[v[i].x]++; count[v[i].y]++; count[v[i].z]++; count[v[i].w]++; }

void doDCount(size_t *dcount, unsigned char *v, size_t i, int mask) {
	if (i == 0) dcount[(((mask + 1) >> 1) + v[i]) & mask]++;
	else dcount[(((mask + 1) >> 1) + v[i] - v[i - 1]) & mask]++;
}
void doDCount(size_t *dcount, unsigned short *v, size_t i, int mask) {
	if (i == 0) dcount[(((mask + 1) >> 1) + v[i]) & mask]++;
	else dcount[(((mask + 1) >> 1) + v[i] - v[i - 1]) & mask]++;
}
void doDCount(size_t *dcount, uchar4 *v, size_t i, int mask) {
	if (i == 0)
	{
		dcount[(((mask + 1) >> 1) + v[i].x) & mask]++;
		dcount[(((mask + 1) >> 1) + v[i].y) & mask]++;
		dcount[(((mask + 1) >> 1) + v[i].z) & mask]++;
		dcount[(((mask + 1) >> 1) + v[i].w) & mask]++;
	}
	else
	{
		dcount[(((mask + 1) >> 1) + v[i].x - v[i - 1].x) & mask]++;
		dcount[(((mask + 1) >> 1) + v[i].y - v[i - 1].y) & mask]++;
		dcount[(((mask + 1) >> 1) + v[i].z - v[i - 1].z) & mask]++;
		dcount[(((mask + 1) >> 1) + v[i].w - v[i - 1].w) & mask]++;
	}
}
void doDCount(size_t *dcount, ushort4 *v, size_t i, int mask) {
	if (i == 0)
	{
		dcount[(((mask + 1) >> 1) + v[i].x) & mask]++;
		dcount[(((mask + 1) >> 1) + v[i].y) & mask]++;
		dcount[(((mask + 1) >> 1) + v[i].z) & mask]++;
		dcount[(((mask + 1) >> 1) + v[i].w) & mask]++;
	}
	else
	{
		dcount[(((mask + 1) >> 1) + v[i].x - v[i - 1].x) & mask]++;
		dcount[(((mask + 1) >> 1) + v[i].y - v[i - 1].y) & mask]++;
		dcount[(((mask + 1) >> 1) + v[i].z - v[i - 1].z) & mask]++;
		dcount[(((mask + 1) >> 1) + v[i].w - v[i - 1].w) & mask]++;
	}
}

template <typename T>
void printSize(T *vol, cudaExtent volumeSize, unsigned int components)
{
	volumeSize.depth *= components;
	// get maximum value and entropy
	size_t *count = new size_t[65536];
	size_t *dcount = new size_t[65536];
	for (unsigned int i = 0; i < 65536; i++)
		count[i] = dcount[i] = 0;
	int max_val = 0;
	int mask = 0;
	for (size_t i = 0; i < (volumeSize.width * volumeSize.height * volumeSize.depth); i++)
	{
		mask = std::max(mask, (int)getMax(vol[i]));
	}
	{
		int tmp = mask;
		mask = 1;
		while (mask < tmp) mask = ((mask + 1) << 1) - 1;
	}
	for (size_t i = 0; i < (volumeSize.width * volumeSize.height * volumeSize.depth); i++)
	{
		max_val = std::max(max_val, (int)getMax(vol[i]));
		doCount(count, vol, i);
		doDCount(dcount, vol, i, mask);
	}
	double ent = 0.0;
	double dent = 0.0;
	for (unsigned int i = 0; i < 65536; i++)
	{
		if (count[i] > 0) ent -= (double)count[i] * log((double)count[i] / (double)(volumeSize.width) / (double)(volumeSize.height) / (double)(volumeSize.depth)) / log(2.0);
		if (dcount[i] > 0) dent -= (double)dcount[i] * log((double)dcount[i] / (double)(volumeSize.width) / (double)(volumeSize.height) / (double)(volumeSize.depth)) / log(2.0);
	}
	delete[] count;
	delete[] dcount;
	printf("Maximum value: %d\n", max_val);
	printf("Entropy: %e\n", (ent / (double)(volumeSize.width) / (double)(volumeSize.height) / (double)(volumeSize.depth)));
	printf("Differential entropy: %e\n", (dent / (double)(volumeSize.width) / (double)(volumeSize.height) / (double)(volumeSize.depth)));
}

template void denoiseVolume<unsigned char>(unsigned char *vol, cudaExtent &volumeSize, int denoise);
template void denoiseVolume<unsigned short>(unsigned short *vol, cudaExtent &volumeSize, int denoise);

template void quantizeVolume<unsigned char, unsigned long long, uint, uchar>(unsigned char *vol, cudaExtent &volumeSize, int lossy, bool bruteForce);
template void quantizeVolume<uchar4, unsigned long long, uint4, uchar4>(uchar4 *vol, cudaExtent &volumeSize, int lossy, bool bruteForce);
template void quantizeVolume<unsigned short, unsigned long long, unsigned long long, unsigned short>(unsigned short *vol, cudaExtent &volumeSize, int lossy, bool bruteForce);
template void quantizeVolume<ushort4, unsigned long long, ulonglong4, ushort4>(ushort4 *vol, cudaExtent &volumeSize, int lossy, bool bruteForce);

template void resampleVolume<unsigned char>(unsigned char *vol, unsigned char *out, cudaExtent &volumeSize, cudaExtent &resampleSize);
template void resampleVolume<unsigned short>(unsigned short *vol, unsigned short *out, cudaExtent &volumeSize, cudaExtent &resampleSize);
template void resampleVolume<uchar4>(uchar4 *vol, uchar4 *out, cudaExtent &volumeSize, cudaExtent &resampleSize);
template void resampleVolume<ushort4>(ushort4 *vol, ushort4 *out, cudaExtent &volumeSize, cudaExtent &resampleSize);

template void expandVolume<unsigned char>(unsigned char *vol, unsigned char *out, cudaExtent &volumeSize, cudaExtent &resampleSize);
template void expandVolume<unsigned short>(unsigned short *vol, unsigned short *out, cudaExtent &volumeSize, cudaExtent &resampleSize);

template unsigned char* calculateGradients<unsigned char>(unsigned char *vol, cudaExtent &volumeSize, unsigned int &components);
template unsigned short* calculateGradients<unsigned short>(unsigned short *vol, cudaExtent &volumeSize, unsigned int &components);

#ifdef LIBPNG_SUPPORT
template unsigned char* loadPngFiles<unsigned char>(char *filename, cudaExtent &volumeSize, float3 &scale, int start, int end, int clip_x0, int clip_x1, int clip_y0, int clip_y1, float scale_png, bool clip_zero);
template unsigned short* loadPngFiles<unsigned short>(char *filename, cudaExtent &volumeSize, float3 &scale, int start, int end, int clip_x0, int clip_x1, int clip_y0, int clip_y1, float scale_png, bool clip_zero);
#endif

template unsigned char *loadRawFile<unsigned char>(char *filename, size_t size, float3 &scale, int raw_skip);
template unsigned short *loadRawFile<unsigned short>(char *filename, size_t size, float3 &scale, int raw_skip);

template void printSize(unsigned char* vol, cudaExtent volumeSize, unsigned int components);
template void printSize(unsigned short* vol, cudaExtent volumeSize, unsigned int components);
template void printSize(uchar4* vol, cudaExtent volumeSize, unsigned int components);
template void printSize(ushort4* vol, cudaExtent volumeSize, unsigned int components);
