cmake_minimum_required(VERSION 3.22)
project(npy_utils)

set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(savedata
        npy_utils.hpp
        npy_utils.cpp
)
