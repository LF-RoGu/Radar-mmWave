#ifndef RADAR_H
#define RADAR_H

#include <cstdint>
#include <vector>
#include <string>
#include <iostream>

const int numBytes = 100 * 40;

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
