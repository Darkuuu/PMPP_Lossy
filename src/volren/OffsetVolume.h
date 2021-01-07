// Copyright (c) 2016 Stefan Guthe / GCC / TU-Darmstadt. All rights reserved. 
// Use of this source code is governed by the BSD 3-Clause license that can be
// found in the LICENSE file.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "CudaIntrinsics.h"

namespace OffsetVolumePrivate
{
	template <typename AT>
	__device__ __forceinline__ AT getAddress(AT *offsetVolume, const int entrySize, const int indirect, const uint3 blk_coord, const uint3 &blkExtent);

	template <typename AT>
	__device__ __forceinline__ AT getAddressFull(AT *offsetVolume, const int entrySize, const int indirect, const AT tag, const uint3 &blkExtent);

	template <typename AT>
	__device__ __forceinline__ AT getTag(AT *offsetVolume, const int entrySize, const int indirect, const uint3 blk_coord, const uint3 &blkExtent);

	template <typename AT>
	__device__ __forceinline__ AT getAddress(AT *offsetVolume, const int entrySize, const int indirect, AT tag, const uint3 &blkExtent);

	template <>
	__device__ __forceinline__ uint getAddress(unsigned int *offsetVolume, const int entrySize, const int indirect, const uint3 blk_coord, const uint3 &blkExtent)
	{
		uint idx = blk_coord.x + (blk_coord.y + blk_coord.z * blkExtent.y) * blkExtent.x;
		uint off;
		// this might exceed 32 bit even if off does fit
		uint64 pos;
		if (indirect > 0)
		{
			pos = (uint64)idx * indirect;
			uint start = (uint)(pos >> 5);
			uint rel = (uint)pos & 0x1f;
			off = ((uint *)offsetVolume)[start] >> rel;
			if (indirect > 32u - rel)
			{
				off += ((uint *)offsetVolume)[start + 1] << (32u - rel);
			}
			off &= 0xffffffffu >> (32u - indirect);
			uint istart = (uint)(((uint64)(blkExtent.x * blkExtent.y * blkExtent.z) * indirect + 31) >> 5);
			pos = (uint64)off * entrySize + (istart << 5);
		}
		else
		{
			pos = (uint64)idx * (uint64)entrySize;
		}
		{
			uint start = (uint)(pos >> 5);
			uint rel = (uint)pos & 0x1f;
			off = ((uint *)offsetVolume)[start] >> rel;
			if (entrySize > 32u - rel)
			{
				off += ((uint *)offsetVolume)[start + 1] << (32u - rel);
			}
			off &= 0xffffffffu >> (32u - entrySize);
		}
		return off;
	}

	template <>
	__device__ __forceinline__ uint64 getAddress(uint64 *offsetVolume, const int entrySize, const int indirect, const uint3 blk_coord, const uint3 &blkExtent)
	{
		uint idx = blk_coord.x + (blk_coord.y + blk_coord.z * blkExtent.y) * blkExtent.x;
		uint64 off;
		uint64 pos;
		if (indirect > 32)
		{
			pos = (uint64)idx * indirect;
			uint64 start = pos >> 6;
			uint64 rel = pos & 0x3f;
			off = ((uint64 *)offsetVolume)[start] >> rel;
			if (indirect > 64u - rel)
			{
				off += ((uint64 *)offsetVolume)[start + 1] << (64u - rel);
			}
			off &= 0xffffffffffffffffull >> (64u - indirect);
			uint64 istart = ((uint64)(blkExtent.x * blkExtent.y * blkExtent.z) * indirect + 63) >> 6;
			pos = off * entrySize + (istart << 6);
		}
		else if (indirect > 0)
		{
			pos = (uint64)idx * indirect;
			uint start = (uint)(pos >> 5);
			uint rel = (uint)pos & 0x1f;
			off = ((uint *)offsetVolume)[start] >> rel;
			if (indirect > 32u - rel)
			{
				off += ((uint *)offsetVolume)[start + 1] << (32u - rel);
			}
			off &= 0xffffffffu >> (32u - indirect);
			uint istart = (uint)(((uint64)(blkExtent.x * blkExtent.y * blkExtent.z) * indirect + 63) >> 6);
			pos = (uint64)off * entrySize + (istart << 6);
		}
		else
		{
			pos = (uint64)idx * entrySize;
		}
		{
			uint64 start = pos >> 6;
			uint64 rel = pos & 0x3f;
			off = ((uint64 *)offsetVolume)[start] >> rel;
			if (entrySize > 64u - rel)
			{
				off += ((uint64 *)offsetVolume)[start + 1] << (64u - rel);
			}
			off &= 0xffffffffffffffffull >> (64u - entrySize);
		}
		return off;
	}

	template <>
	__device__ __forceinline__ uint getAddressFull(unsigned int *offsetVolume, const int entrySize, const int indirect, const uint tag, const uint3 &blkExtent)
	{
		uint idx = tag;
		uint off;
		// this might exceed 32 bit even if off does fit
		uint64 pos;
		if (indirect > 0)
		{
			pos = (uint64)idx * indirect;
			uint start = (uint)(pos >> 5);
			uint rel = (uint)pos & 0x1f;
			off = ((uint *)offsetVolume)[start] >> rel;
			if (indirect > 32u - rel)
			{
				off += ((uint *)offsetVolume)[start + 1] << (32u - rel);
			}
			off &= 0xffffffffu >> (32u - indirect);
			uint istart = (uint)(((uint64)(blkExtent.x * blkExtent.y * blkExtent.z) * indirect + 31) >> 5);
			pos = (uint64)off * entrySize + (istart << 5);
		}
		else
		{
			pos = (uint64)idx * entrySize;
		}
		{
			uint start = (uint)(pos >> 5);
			uint rel = (uint)pos & 0x1f;
			off = ((uint *)offsetVolume)[start] >> rel;
			if (entrySize > 32u - rel)
			{
				off += ((uint *)offsetVolume)[start + 1] << (32u - rel);
			}
			off &= 0xffffffffu >> (32u - entrySize);
		}
		return off;
	}

	template <>
	__device__ __forceinline__ uint64 getAddressFull(uint64 *offsetVolume, const int entrySize, const int indirect, const uint64 tag, const uint3 &blkExtent)
	{
		uint idx = tag;
		uint64 off;
		uint64 pos;
		if (indirect > 32)
		{
			pos = (uint64)idx * indirect;
			uint64 start = pos >> 6;
			uint64 rel = pos & 0x3f;
			off = ((uint64 *)offsetVolume)[start] >> rel;
			if (indirect > 64u - rel)
			{
				off += ((uint64 *)offsetVolume)[start + 1] << (64u - rel);
			}
			off &= 0xffffffffffffffffull >> (64u - indirect);
			uint64 istart = ((uint64)(blkExtent.x * blkExtent.y * blkExtent.z) * indirect + 63) >> 6;
			pos = off * entrySize + (istart << 6);
		}
		else if (indirect > 0)
		{
			pos = (uint64)idx * indirect;
			uint start = (uint)(pos >> 5);
			uint rel = (uint)pos & 0x1f;
			off = ((uint *)offsetVolume)[start] >> rel;
			if (indirect > 32u - rel)
			{
				off += ((uint *)offsetVolume)[start + 1] << (32u - rel);
			}
			off &= 0xffffffffu >> (32u - indirect);
			uint istart = (uint)(((uint64)(blkExtent.x * blkExtent.y * blkExtent.z) * indirect + 63) >> 6);
			pos = (uint64)off * entrySize + (istart << 6);
		}
		else
		{
			pos = (uint64)idx * entrySize;
		}
		{
			uint64 start = pos >> 6;
			uint64 rel = pos & 0x3f;
			off = ((uint64 *)offsetVolume)[start] >> rel;
			if (entrySize > 64u - rel)
			{
				off += ((uint64 *)offsetVolume)[start + 1] << (64u - rel);
			}
			off &= 0xffffffffffffffffull >> (64u - entrySize);
		}
		return off;
	}

	template <>
	__device__ __forceinline__ uint getTag(unsigned int *offsetVolume, const int entrySize, const int indirect, const uint3 blk_coord, const uint3 &blkExtent)
	{
		uint idx = blk_coord.x + (blk_coord.y + blk_coord.z * blkExtent.y) * blkExtent.x;
		uint off;
		// this might exceed 32 bit even if off does fit
		uint64 pos;
		if (indirect > 0)
		{
			pos = (uint64)idx * indirect;
			uint start = (uint)(pos >> 5);
			uint rel = (uint)pos & 0x1f;
			off = ((uint *)offsetVolume)[start] >> rel;
			if (indirect > 32u - rel)
			{
				off += ((uint *)offsetVolume)[start + 1] << (32u - rel);
			}
			off &= 0xffffffffu >> (32u - indirect);
			return off;
		}
		else
		{
			pos = (uint64)idx * (uint64)entrySize;
			uint start = (uint)(pos >> 5);
			uint rel = (uint)pos & 0x1f;
			off = ((uint *)offsetVolume)[start] >> rel;
			if (entrySize > 32u - rel)
			{
				off += ((uint *)offsetVolume)[start + 1] << (32u - rel);
			}
			off &= 0xffffffffu >> (32u - entrySize);
			return off;
		}
	}

	template <>
	__device__ __forceinline__ uint64 getTag(uint64 *offsetVolume, const int entrySize, const int indirect, const uint3 blk_coord, const uint3 &blkExtent)
	{
		uint idx = blk_coord.x + (blk_coord.y + blk_coord.z * blkExtent.y) * blkExtent.x;
		uint64 off;
		uint64 pos;
		if (indirect > 32)
		{
			pos = (uint64)idx * indirect;
			uint64 start = pos >> 6;
			uint64 rel = pos & 0x3f;
			off = ((uint64 *)offsetVolume)[start] >> rel;
			if (indirect > 64u - rel)
			{
				off += ((uint64 *)offsetVolume)[start + 1] << (64u - rel);
			}
			off &= 0xffffffffffffffffull >> (64u - indirect);
			return off;
		}
		else if (indirect > 0)
		{
			pos = (uint64)idx * indirect;
			uint start = (uint)(pos >> 5);
			uint rel = (uint)pos & 0x1f;
			off = ((uint *)offsetVolume)[start] >> rel;
			if (indirect > 32u - rel)
			{
				off += ((uint *)offsetVolume)[start + 1] << (32u - rel);
			}
			off &= 0xffffffffu >> (32u - indirect);
			return off;
		}
		else
		{
			pos = (uint64)idx * entrySize;
			uint64 start = pos >> 6;
			uint64 rel = pos & 0x3f;
			off = ((uint64 *)offsetVolume)[start] >> rel;
			if (entrySize > 64u - rel)
			{
				off += ((uint64 *)offsetVolume)[start + 1] << (64u - rel);
			}
			off &= 0xffffffffffffffffull >> (64u - entrySize);
			return off;
		}
	}

	template <>
	__device__ __forceinline__ uint getAddress(unsigned int *offsetVolume, const int entrySize, const int indirect, const uint tag, const uint3 &blkExtent)
	{
		if (indirect == 0) return tag;
		uint off;
		// this might exceed 32 bit even if off does fit
		uint64 pos;
		{
			uint istart = (uint)(((uint64)(blkExtent.x * blkExtent.y * blkExtent.z) * indirect + 31) >> 5);
			pos = (uint64)tag * entrySize + (istart << 5);
		}
		{
			uint start = (uint)(pos >> 5);
			uint rel = (uint)pos & 0x1f;
			off = ((uint *)offsetVolume)[start] >> rel;
			if (entrySize > 32u - rel)
			{
				off += ((uint *)offsetVolume)[start + 1] << (32u - rel);
			}
			off &= 0xffffffffu >> (32u - entrySize);
		}
		return off;
	}

	template <>
	__device__ __forceinline__ uint64 getAddress(uint64 *offsetVolume, const int entrySize, const int indirect, const uint64 tag, const uint3 &blkExtent)
	{
		if (indirect == 0) return tag;
		uint64 off;
		uint64 pos;
		if (indirect > 32)
		{
			uint64 istart = ((uint64)(blkExtent.x * blkExtent.y * blkExtent.z) * indirect + 63) >> 6;
			pos = tag * entrySize + (istart << 6);
		}
		else
		{
			uint istart = (uint)(((uint64)(blkExtent.x * blkExtent.y * blkExtent.z) * indirect + 63) >> 6);
			pos = (uint64)tag * entrySize + (istart << 6);
		}
		{
			uint64 start = pos >> 6;
			uint64 rel = pos & 0x3f;
			off = ((uint64 *)offsetVolume)[start] >> rel;
			if (entrySize > 64u - rel)
			{
				off += ((uint64 *)offsetVolume)[start + 1] << (64u - rel);
			}
			off &= 0xffffffffffffffffull >> (64u - entrySize);
		}
		return off;
	}
}

// always needs address type as template parameter (either uint or uint64)
template <typename AT>
class OffsetVolume
{
protected:
	// constructor
	AT *m_data;
	int m_entrySize;
	int m_indirect;
	uint3 m_blkExtent;
public:
	// this one is allowed since we store the compacted data inside the renderer class
	// the only drawback is that we require one instance for 32 and one instance for 64 bit
	OffsetVolume() { m_data = NULL; }
	OffsetVolume(AT *data, int entrySize, int indirect, uint3 blkExtent) : m_data(data), m_entrySize(entrySize), m_indirect(indirect), m_blkExtent(blkExtent) {}
	OffsetVolume(const OffsetVolume &a) : m_data(a.m_data), m_entrySize(a.m_entrySize), m_indirect(a.m_indirect), m_blkExtent(a.m_blkExtent) {}
	~OffsetVolume() { }
	void destroy() { if (m_data != NULL) checkCudaErrors(cudaFree(m_data)); m_data = NULL; }

#ifdef ADDRESS_TAG_ID
	__device__ __forceinline__ AT getAddress(const uint3 blk_coord)
	{
		return OffsetVolumePrivate::getAddress<AT>(m_data, m_entrySize, m_indirect, blk_coord, m_blkExtent);
	}
#endif
#ifdef BINDEX_TAG_ID
	__device__ __forceinline__ AT getTag(const uint3 blk_coord)
	{
		return blk_coord.x + (blk_coord.y + blk_coord.z * m_blkExtent.y) * m_blkExtent.x;
	}
	__device__ __forceinline__ AT getAddress(const AT tag)
	{
		return OffsetVolumePrivate::getAddressFull<AT>(m_data, m_entrySize, m_indirect, tag, m_blkExtent);
	}
#endif
#ifdef INDIRECT_TAG_ID
	__device__ __forceinline__ AT getTag(const uint3 blk_coord)
	{
		return OffsetVolumePrivate::getTag<AT>(m_data, m_entrySize, m_indirect, blk_coord, m_blkExtent);
	}
	__device__ __forceinline__ AT getAddress(const AT tag)
	{
		return OffsetVolumePrivate::getAddress<AT>(m_data, m_entrySize, m_indirect, tag, m_blkExtent);
	}
#endif
};
