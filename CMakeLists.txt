cmake_minimum_required(VERSION 3.6)
project(training)

FIND_PACKAGE(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp)
add_executable(training ${SOURCE_FILES})

target_link_libraries(training ${OpenCV_LIBS})