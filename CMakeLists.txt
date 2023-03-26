PROJECT(superpoint LANGUAGES CXX)
cmake_minimum_required(VERSION 3.23 FATAL_ERROR)

LIST(APPEND CMAKE_PREFIX_PATH "/opt/libtorch")
# SET(CMAKE_CUDA_ARCHITECTURES all)
find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)
find_package(OpenCV REQUIRED)
# find_package(CUDAToolkit)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

SET(CMAKE_CXX_FLAGS, "${CMAKE_CXX_FLAGS} -g")
add_executable(superpoint_script_test
    src/superpoint_script.cc
    src/superpoint_script_test.cc)
target_include_directories(superpoint_script_test PRIVATE
    ${OpenCV_INCLUDE_DIRS})
target_link_libraries(superpoint_script_test PUBLIC
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
    ${TORCHVISION_LIBRARIES}
    gflags
    TorchVision::TorchVision
    glog)
# target_link_libraries(superpoint_script_test PUBLIC TorchVision::TorchVision)
set_property(TARGET superpoint_script_test PROPERTY CXX_STANDARD 14)
