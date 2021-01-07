#
# 
# Name the project you can choose any name you want here
PROJECT(MyProject) 


# Check requirements
CMAKE_MINIMUM_REQUIRED(VERSION 2.6.3)

# set the compiler flags
SET(CMAKE_CXX_COMPILER g++)
SET(CMAKE_C_COMPILER gcc)


SET(CMAKE_CXX_FLAGS "-fPIC -Wall -O3 -Wfatal-errors -fstrict-aliasing -m64 -std=c++11 -fopenmp")


# Add needed packages (e.g. libraries)
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(OpenGL REQUIRED)
FIND_PACKAGE(GLUT REQUIRED)
FIND_PACKAGE(GLEW REQUIRED)
FIND_PACKAGE(ZLIB REQUIRED)
FIND_PACKAGE(JPEG REQUIRED)
FIND_PACKAGE(PNG REQUIRED)

# Define where your executables should be put
set(EXECUTABLE_OUTPUT_PATH ${MyProject_BINARY_DIR})


# Define where your build libraries should be put
set(LIBRARY_OUTPUT_PATH ${MyProject_BINARY_DIR}/lib)



# Include directories and Link directories
# you should add all your folders here, as this makes copying and reusing your code easier
# pay attention to QT folders, as you might need to add the binary folder as well
INCLUDE_DIRECTORIES(
			${CUDA_INCLUDE_DIRS}
			${OPENGL_INCLUDE_DIRS}
			${GLUT_INCLUDE_DIRS}
			${GLEW_INCLUDE_DIRS}
			${ZLIB_INCLUDE_DIRS}
			${JPEG_INCLUDE_DIRS}
			${PNG_INCLUDE_DIRS}
			${MyProject_SOURCE_DIR}/src/common/inc
			${MyProject_SOURCE_DIR}/src/volren
)


LINK_DIRECTORIES(
)


# CUDA stuff
# set emulation to on for debugging cuda code
SET(CUDA_BUILD_EMULATION OFF)
# set to on to print all the commands used for compiling of cuda code
SET(CUDA_VERBOSE_BUILD ON)
# set to on to print memory usage during build
SET(CUDA_BUILD_CUBIN ON)
SET(CUDA_NVCC_FLAGS "-gencode=arch=compute_52,code=sm_52;-use_fast_math")#;-std=c++11")

# Tell CMake to process the sub-directories
ADD_SUBDIRECTORY(src)


