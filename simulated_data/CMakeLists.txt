cmake_minimum_required(VERSION 2.8)

add_definitions(-std=c++11)
#set(CMAKE_CXX_FLAGS_RELEASE "-Wall -std=c++11 -O2")  
#set(CMAKE_CXX_FLAGS_DEBUG "-Wall -g -std=ctool11 -O2")

project(SimulatedData)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Boost REQUIRED COMPONENTS system thread)
include_directories(${Boost_INCLUDE_DIRS})

set(EIGEN3_INCLUDE_DIRS "/usr/include/eigen3")
include_directories(${EIGEN3_INCLUDE_DIRS})

set(BOOST_THREADPOOL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/boost_tool")
include_directories(BOOST_THREADPOOL_INCLUDE_DIR)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/srcs)

file(GLOB xsrcs srcs/*.hpp srcs/*.cpp)

add_executable(simulated main.cpp ${xsrcs})
target_link_libraries(simulated pthread ${OpenCV_LIBS} ${Boost_LIBRARIES})
