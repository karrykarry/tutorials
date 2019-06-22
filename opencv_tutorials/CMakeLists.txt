cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(opencv_tutorials)

find_package(OpenCV 3 REQUIRED)

SET(CMAKE_CXX_FLAGS "-std=c++11 -O2 -g -Wall ${CMAKE_CXX_FLAGS}")

add_executable (image_show image_show.cpp)
target_link_libraries (image_show ${OpenCV_LIBRARIES})

