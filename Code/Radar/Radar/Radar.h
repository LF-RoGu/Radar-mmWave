#ifndef RADAR_H
#define RADAR_H

#include <cstdint>
#include <vector>
#include <string>
#include <iostream>

// Define reading modes
#define CONTINUOUS_READING 1       // 1 for continuous, 0 for limited frames
#define FIXED_FRAME_COUNT 10       // Set the number of frames to read if continuous is disabled


struct FrameHeader {
    uint32_t syncWord;
    uint32_t version;
    uint32_t totalPacketLen;
    uint32_t platform;
    uint32_t frameNumber;
    uint32_t timeCpuCycles;
    uint32_t numDetectedObj;
    uint32_t numTLVs;
    uint32_t subFrameNumber;
};

struct TlvHeader {
    uint32_t type;
    uint32_t length;
};

struct DetectedObject {
    float range;
    float doppler;
    float peakVal;
    float x;
    float y;
    float z;
};

extern std::vector<uint8_t> values;

#endif // RADAR_H
