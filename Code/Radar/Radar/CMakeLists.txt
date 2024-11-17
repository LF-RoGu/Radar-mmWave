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
include_directories(include radar_sensor/sensor_data/lib radar_sensor)

# Add individual source files and headers
set(SOURCES
    "Radar.cpp" "Radar.h"
    "radar_sensor/IWR6843.cpp" "radar_sensor/IWR6843.h"
    "radar_sensor/sensor_data/UARTFrame.cpp" "radar_sensor/sensor_data/UARTFrame.h"
)

# Add the executable for the project
add_executable(Radar ${SOURCES} "radar_sensor/sensor_data/lib/EndianUtils.h" "radar_sensor/sensor_data/lib/EndianUtils.cpp")

# Add any required libraries here if needed
# target_link_libraries(Radar <library_name>)
