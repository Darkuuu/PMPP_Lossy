# volumeRenderer.cpp
- before main:
    - several helper functions
    - rendering functions for OpenGL
- main method
    - begins ~ line 1520
    - line 1520 - 1964 -> command line flag checks
    - line 1982 - 2007 -> select renderer based on compression flag
        - if compression is used RendererRBUC8x8 is selected (in 8, 16, 32 or 64 version respectively)

# RendererRBUC8x8.h / .cpp
- Handles rendering of compressed data
- written in CUDA
- uses CompressRBUC.h
- in cpp line 1215 - 1266 -> different implementations for compress function which calls compress_internal1 and compress_internal2
    - compress_internal2 calls compressRBUC8x8
- compressInternal2 in line 1082

### Transformation functions
- transformGradient line 813
- transformHaarInternal line 832
- transformHaar line 935 / 941 / 947 / 966
- transformSubMin line 985
- transformSubMax line 1001
- swizzleWavelet line 1017
- swizzleRegular line 1046

### Blocking
- the raw data is divided in blocks from line 186 - line 292
    - groups of 3 for loops each iterate through 3D data
    - blocks are written to raw_dat which is an array with length 64
    - three iteration variables always are converted to one dimensional index for array access
- raw_dat is then given to the internal compress function (RendererRBUC8x8::compress) which begins in line 1224

### Predict
- "predict" functions from line 734 - line 808
- two functions for 1d values, two for 4d vectors
    - each in unsigned short and unsigned char implementation
- all work in exactly the same way
- Use neighboring pixel values for prediction
    - idx - 1 = pixel directly before (left pixel (x - 1))
    - idx - 4 = pixel directly above (top pixel (y - 1))
    - idx - 5 = pixel directly diagonal left above (top left pixel (x - 1, y - 1))
    - idx - 16 = pixel directly in front (front pixel (z - 1))
    - idx - 17 = pixel directly in front left (front left pixel (x - 1, z - 1))
    - idx - 20 = pixel directly in front above (front top pixel (y - 1, z - 1))
    - idx - 21 = pixel directly in front diagonal left above (front top left pixel (x - 1, y - 1, z - 1))