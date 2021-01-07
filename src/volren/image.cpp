// Copyright (c) 2016 Stefan Guthe / GCC / TU-Darmstadt. All rights reserved. 
// Use of this source code is governed by the BSD 3-Clause license that can be
// found in the LICENSE file.

#include "image.h"

void Image::CopyImage(const Image &source)
{
	SetWidth(source.GetWidth());
	SetHeight(source.GetHeight());
	SetComponents(source.GetComponents());
	for (unsigned int y = 0; y < GetHeight(); y++)
	{
		for (unsigned int x = 0; x < GetWidth(); x++)
		{
			for (unsigned int c = 0; c < GetComponents(); c++)
			{
				VSetValue(x, y, c, source.VGetValue(x, y, c));
			}
		}
	}
}

