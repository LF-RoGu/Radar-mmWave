#ifndef SENSORDATASTR_H
#define SENSORDATASTR_H

#include <cstdint>
#include <vector>
#include <iostream>

#define CONTINUOUS_READING 1       // 1 for continuous, 0 for limited frames
#define FIXED_FRAME_COUNT 10       // Number of frames to read if continuous is disabled

struct FrameHeaderData {
    uint16_t magicWord_u16;
    uint32_t version_u32;
    uint32_t totalPacketLength_u32;
    uint32_t platform_u32;
    uint32_t frameNumber_u32;
    uint32_t timeCpuCycles_u32;
    uint32_t numDetectedObj_u32;
    uint32_t numTLVs_u32;
    uint32_t subFrameNumber_u32;
};

struct TLVHeaderData {
    uint32_t type_u32;
    uint32_t length_u32;
};

// Additional structs for various types of detected data
struct DetectedPoints {
    float x_f;
    float y_f;
    float z_f;
    float doppler_f;
};

#endif // SENSORDATASTR_H
