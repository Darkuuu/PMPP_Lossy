// Copyright (c) 2016 Stefan Guthe / GCC / TU-Darmstadt. All rights reserved. 
// Use of this source code is governed by the BSD 3-Clause license that can be
// found in the LICENSE file.

#pragma once

// these are shorter
#ifndef int64
typedef long long int64;
#endif
#ifndef uint64
typedef unsigned long long uint64;
#endif

// regular tag id uses block address in all cases (disable all defines for default)
//#define ADDRESS_TAG_ID
// no-memory cache tag (reduces hit rate for multiple references but overall a lot faster)
#define BINDEX_TAG_ID
// indirect index or direct address (halfway tag id, faster for indirect only)
//#define INDIRECT_TAG_ID
// dual cache tag (requires additional entry for multiple references, currently not supported)
//#define DUAL_TAG_ID
