# CMakeLists.txt for "Radar" project

# Define the minimum CMake version and project name
cmake_minimum_required(VERSION 3.12)
project(Radar)

# Set C++ standard to C++20 if CMake version is greater than 3.12
if (CMAKE_VERSION VERSION_GREATER 3.12)
  set(CMAKE_CXX_STANDARD 20)
endif()

# Include directories for headers
include_directories(${CMAKE_SOURCE_DIR})

# Add individual source files and headers
set(SOURCES
    "Radar.cpp" "Radar.h"
    "SerialPort.cpp" "SerialPort.h"
    "Decode.cpp" "Decode.h"
    "ReadData.cpp" "ReadData.h"
    "App.cpp" "App.h"
    "RadarFrame.cpp" "RadarFrame.h")

# Add the executable for the project
add_executable(Radar ${SOURCES})

# Add any required libraries here if needed
# target_link_libraries(Radar <library_name>)
