CMAKE_MINIMUM_REQUIRED(VERSION 2.8.9)
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)
CUDA_ADD_EXECUTABLE( main blind.cu buildPyramid.cu coarseToFine.cu conv.cu dec.cu GetGradK.cu GPUConvolve.cu gradTVcc.cu helper.cu KernelProjection.cu main.cu MaxFinder.cu NonConvexOptimization.cu scalarMult.cu SubtractAB.cu )

TARGET_LINK_LIBRARIES(main ${OpenCV_LIBS} ${CUDA_LIBS})
CUDA_ADD_CUBLAS_TO_TARGET(main)

