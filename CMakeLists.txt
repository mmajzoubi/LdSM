cmake_minimum_required(VERSION 2.8.9)
project(stream)

#Bring the headers, such as Student.h into the project
include_directories(include)

#Can manually add the sources using the set command as follows:
#set(SOURCES src/mainapp.cpp src/Student.cpp)
SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++11 -pthread -Ofast")

#However, the file(GLOB...) allows for wildcard additions:
file(GLOB SOURCES "src/*.cpp")

find_package (Threads)
add_executable(stream ${SOURCES})
target_link_libraries (stream ${CMAKE_THREAD_LIBS_INIT})
