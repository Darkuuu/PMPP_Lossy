Copyright (c) 2016 Stefan Guthe / GCC / TU-Darmstadt. All rights reserved. 
Use of this source code is governed by the BSD 3-Clause license that can be
found in the LICENSE file.

If you find the source code provided useful in any way, please send us a short
email to stefan.guthe@gris.informatik.tu-darmstadt.de
If not, please send us comments as to how to improve this release.

1. Build instructions for cmake (linux and Windows)

  Building the project requires
    - cmake 2.6.3 or newer
    - gcc
    - CUDA 8.0
    - OpenGL
    - GLUT
    - GLEW

  Optional dependencies
    - zlib (enabled by default)
    - libjpeg (enabled by default)
    - libpng (enabled by default)
    - v3 (disabled by default)
    Each dependencie can be enabled/disabled in "config.h"
    Library dependencies have a corresponding find_package call "CMakeLists.txt"
    that needs to be consistent with "config.h"

  Inside the "volcomp" folder execute the following commands (linux only)
    > mkdir build
    > cd build
    > cmake ..
    > make -j12

2. Build instructions for Visual Studio

  Building the project requires
    - MS Visual Studio 2013
    - CUDA 8.0
    - OpenGL
    - GLUT (contained within the archive)
    - GLEW (contained within the archive)

  Optional dependencies (source code required)
    - zlib (disabled by default)
    - libjpeg (disabled by default)
    - libpng (disabled by default)
    - v3 (disabled by default)
    Each dependencie can be enabled/disabled in "config.h"
    If a dependency is enabled, the source code needs to be placed into the
    corresponding directory (see below).
    If a dependencie is disabled, the corresponding project needs to be removed
    from the volcomp solution before building (libraries only).

  Source code for optional dependencies
    - http://zlib.net/ (zlib)
      extract all source files to "volcomp/zlib/"

    - http://libjpeg.sourceforge.net/ (libjpeg)
      extract all source files to "volcomp/libjpeg/"

    - http://www.libpng.org/pub/png/ (libpng)
      extract all source files to "volcomp/libpng/"

    - https://sourceforge.net/projects/volren/ (v^3)
      extract "codebase.h", "ddsbase.cpp" and "ddsbase.h" to "volcomp/src/v3/"

3. Usage

  Simply call the volren application with the volume file to load and display
  specified by the "-volume" command line option
    > volren -volume=Bucky-32x32x32_8.ddv

  File formats supported (if support is enabled)
    - raw 8 bit and 16 bit
      Use -xsize=<width>, -ysize=<height> and -zsize=<depth> or -size=<dim> for
      cubic volumes to specify dimensions of the data.
      Select between 8 and 16 bit by using -raw=1 (8 bit) and -raw=2 (16 bit)
      Use -raw_components=<comp> for multiple components

    - dat file
      16 bit only file format with simple header and raw data.
      Single component format only.
      See "https://www.cg.tuwien.ac.at/research/vis/datasets/" for datasets.

    - png files (slices)
      8 or 16 bit, single and multiple components.
      Can be used for loading the visible human for example:
      See "https://www.nlm.nih.gov/research/visible/visible_human.html"
      Uses printf style formatting to insert slice number into filename.
      Specify using the following options:
      -png -start=<first_slice_numer> -end=<last_slice_number>

    - pvm file
      File format of v^3 volume renderer (8 or 16 bit multiple components).
      Stores physical extent of voxel for non-cubic voxels.
      See "http://lgdv.cs.fau.de/External/vollib/" for datasets.

    - ddv file
      Internal file format with optional compression.
      8 or 16 bit, single and multiple components.
      Stores physical extent of voxel for non-cubic voxels.
      See project page for datasets.

  All files can be compressed with gzip for storage and read directly if libz
  support is enabled.

  In addition, you can use any of the following options

    -16bit
      Upload 16 bit volume data onto the GPU.
      Only allowed for volume data containing more that 8 bit per component.

    -8bit
      Upload 8 bit volume data onto the GPU.
      Can be combined with -linear=<mode> if the original data contains more
      than 8 bit per component.

    -linear=<mode>
      Conversion mode to 8 bit volume data
      0: Maximum entropy conversion
         out = map(in) with map(x) <= map(x + 1) and maxmial entropy of out
      1: linear mapping
         out = floor(255.0 * in / max_in + 0.5)
      2: exponential mapping
         out = floor(exp(log(255.0) * log(in) / log(max_in) + 0.5)
      3: logarithmic mapping
         out = floor(256.0 * (1.0 - exp(-8.0 * log(2.0) * in / max_in) + 0.5)

    -xscale=<extent_x> -yscale=<extent_y> -zscale=<extent_z>
      Specify the physical extent of a voxel (float in mm)

    -xresample=<width> -yresample=<height> -zresample=<depth>
      Resample (scale) volume data

    -rgba -combined
      Both options specify that the volume data is stored as a vector of 4
      components rather than 4 consecutive volumes.

    -split
      Volume data with <n> components stored as <n> consecutive volumes.

    -swidth=<width> -sheight=<height>
      Specify the size of the window at startup.

    -fixed
      Fixed sample distance at 0.01 where 1.0 is the largest extent of the
      volume data.
      If not specified, the sampling distance is the minimum of the extent
      of a single voxel in x, y, z but always smaller than the 0.01 as above.

    -gradient
      Calculate high quality gradients and attach the result to the original
      volume data.
      Only valid for data containing only a single component.
      The result will be a new volume containing four components.

    -compression=<mode>
      0: uncompressed texture
      1: RBUC based compression

    -compare
      Display <max_in> and encoded entropy.
      The entropy is the bit per voxel that can be achieved by replacing RBUC
      with arithmetic coding.
      However, a stream using arithmetic coding could not be decompressed on
      the GPU during rendering.

    -denoise=<mode>
      Noise reduction for very noisy CT data.
      0: do nothing
      1: for each row along the x-axis, subtract minimum value of that row.
      2: for each row along the y-axis, subtract minimum value of that row.
      3: for each row along the z-axis, subtract minimum value of that row.
      4: Christmas present noise reduction.
         For each slice it subtracts the maxmimum noise value found outside
         of the actual present but within the radius of the CT scan.

    -lossy=<vq>
      Quantizer setting for data resuction.
      0: no reduction (all unique 4x4x4 blocks present)
      1: reduce to half the number of blocks to half the numer of unique blocks
      2: reduce to 1/4th
      3: reduce to 1/8th
      n: reduce to 1/(2^n)th

    -bruteforce
      Brute force initialization mode for quantizer.
      Increases performance for very large datasets.

  Benchmarking options

    -auto
      Run benchmark of 1000 frames at 512x512 and 100 frames at 1920x1080.
      All camera positions are randomly distributed around the dataset.
      Results are also appended to a file called "logfile.txt" for batch mode.

    -lighting
      Enable lighting at startup.

    -color
      Enable color at startup.
      Together with lighting this will enable non-photorealistic rendering.

    -transfer=<n>
      Select from a set of transfer functions.

    -density=<d>
      Define the density multiplier (default is 0.05)

    -brightness=<b>
      Define the brightness multiplier (default is 1.0)

    -transferScale=<s>
      Define the transfer function scaling (default is 1.0)

    -transferOffset=<o>
      Define the transfer function offset (default is 0.0)

  File conversion

    -export=<volume_file>
      Write the volume to a different file and file type after loading.
      All file types used for loading can also be used for export.
      File type is specified using filename extension.
      If the filename extension is .gz the file will be compressed using libz
      and the volume type is defined by the filename extension that is found
      after removing .gz from the filename.

    -export_version=<version>
      Used for ddv files only.
      0: latest (same as 4)
      1: no compression
      2: RBUC compression only
      3: RBUC + RLE0 (each 0 byte is followed by a one byte repeat counter)
      4: RBUC + arithmetic coding (highest compression rate)

  png loading specific otions

    -scale_png=<scale>
      Up- or downscale png files during reading.

    -clip_x0=<left> -clip_x1=<right> -clip_y0=<top> -clip_y1=<bottom>
      Clip the given number of pixels from every png file during reading
      (prior to any scaling).

    -clip_zero
      Set the color of evry transparant pixel in a png file to (0,0,0).

  Further experimental options

    -empty 
      Create an empty volume instead of loading data use with -size=<dim> or
      -xsize=<width> -ysize=<height> -zsize=<depth>.

    -xoriginal=<width> -yoriginal=<height> -zoriginal=<depth>
      Clips the volume to a cretain size before compression.
      Used for loading raw volume files that were padded during quantization.

    -fast
      Faster volume compression but less efficient if the volume size is not a
      multiple of 4 in all directions.

    -expand
      Expand the volume to be a multiple of 4 in width, heigth and depth

    -histogram
      Dump image file containing the histogram of the volume dataset.

    -basic
      Basic volume rendering algorithm used in CUDA SDK example.
      Only valid when volume compression is deactivated.

    -raw_skip=<n>
      Skip the first <n> slices while reading a raw file.

    -compare_a=<filename> -compare_b=<filename>
      Compare two ktx texture files and calculate the PSNR.

    -learning
      Try to learn the best launch configuration.
      Do not use this option, result is not defined.

  Interaction during rendering

    - left mouse button: rotate around current x- and y-axis.
    - middle mouse button: translate along x and y.
    - right mouse button: translate along z.
    - '+' and '-' to change density (0.01 increments)
    - ']' and '[' to change brightness
    - ';' and ''' to modify transfer function offset
    - '.' and ',' to modify transfer function scale
    - 'f'         toggle filter
    - 'b'         toggle background
    - 'c'         toggle rgba color volume
    - 'l'         toggle lighting (includes gradient magnitude modulation with
                  color volume on)
    - 'ESC'       exit program
    - 'a'         start automatic benchmark mode
    - '0' - '9'   select transfer function
    - 'q'         display current rotation and translation
    - 'w', 'e', 'r', 't', 'u', 'i', 'o', 'p', 'k', 'j'
                  pre-defined camera positions
    - 's'         create a snapshot file with 512x512 pixel
    - 'm'         create a snapshot file with 1024x1024 pixel
    - 'x'         create a snapshot file with 2048x2048 pixel
    - 'z'         create a snapshot file with 4096x4096 pixel
    - '*'         enable and disable interpolation of the transfer function.

