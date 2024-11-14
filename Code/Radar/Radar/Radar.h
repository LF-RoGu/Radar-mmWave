#ifndef RADAR_H
#define RADAR_H

#include <cstdint>
#include <vector>
#include <string>
#include <iostream>

// Define reading modes
#define CONTINUOUS_READING 1       // 1 for continuous, 0 for limited frames
#define FIXED_FRAME_COUNT 10       // Set the number of frames to read if continuous is disabled


extern std::vector<uint8_t> values;

#endif // RADAR_H
