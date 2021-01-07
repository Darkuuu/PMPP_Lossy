# Put all your source files here 
SET(volren_SRCS
  BasicRenderer.cu
  BasicRenderer_8.cu
  BasicRenderer_16.cu
  BasicRenderer_32.cu
  BasicRenderer_64.cu
  CompressRBUC.cpp
  image.cpp
  jpeg_image.cpp
  png_image.cpp
  pnm_image.cpp
  RendererRBUC8x8.cu
  RendererRBUC8x8_8.cu
  RendererRBUC8x8_16.cu
  RendererRBUC8x8_32.cu
  RendererRBUC8x8_64.cu
  volumeReader.cpp
  volumeRender.cpp
)

# this command creates your executable from the files specified in MyProject_SRCS
CUDA_ADD_EXECUTABLE(volren ${volren_SRCS})

# link all necessary libraries here (only the names are needed, the rest is done by CMake)
TARGET_LINK_LIBRARIES(volren ${CUDA_LIBRARIES} png jpeg ${ZLIB_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${GLEW_LIBRARIES})
