PROJECT(superpoint)
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
# set(CMAKE_CXX_STANDARD 14)

LIST(APPEND CMAKE_PREFIX_PATH "/opt/libtorch")
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

add_executable(superpoint_script_test src/superpoint_script_test.cc)
target_include_directories(superpoint_script_test PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(superpoint_script_test 
    ${OpenCV_LIBS} 
    ${TORCH_LIBRARIES}
    gflags
    glog)
set_property(TARGET superpoint_script_test PROPERTY CXX_STANDARD 14)
