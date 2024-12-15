# CMakeList.txt : CMake project for Radar, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("Radar")

# Add include directories for sensor_data and lib
include_directories(
    ${CMAKE_SOURCE_DIR}/radar_sensor/sensor_data
    ${CMAKE_SOURCE_DIR}/radar_sensor/sensor_data/lib
)

# Add source to this project's executable.
# Change the executable name to avoid conflict
add_executable (
    Radar  
    "Radar.cpp" "Radar.h"
    "radar_sensor/IWR6843.cpp" "radar_sensor/IWR6843.h" 
    "radar_sensor/SensorData.cpp" "radar_sensor/SensorData.h" 
    "radar_sensor/sensor_data/UARTFrame.cpp" "radar_sensor/sensor_data/UARTFrame.h" 
    "radar_sensor/sensor_data/lib/EndianUtils.cpp" "radar_sensor/sensor_data/lib/EndianUtils.h" 
    "radar_sensor/sensor_data/lib/SensorDataStr.h"
)

file(COPY ${CMAKE_SOURCE_DIR}/configs/BestRange_30fps.cfg 
     DESTINATION ${CMAKE_BINARY_DIR}/configs)



if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET Radar PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
