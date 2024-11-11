#ifndef SENSORDATASTR_H
#define SENSORDATASTR_H

#include <cstdint>
#include <vector>
#include <iostream>

#pragma once
// Define reading modes
#define CONTINUOUS_READING 1       // 1 for continuous, 0 for limited frames
#define FIXED_FRAME_COUNT 10       // Set the number of frames to read if continuous is disabled


struct FrameHeader {
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

struct TLVHeader {
    uint32_t type_u32;
    uint32_t length_u32;
};

struct DetectedPoints {
    float x_f;
    float y_f;
    float z_f;
    float doppler_f;
};

struct RangeProfile {
    float data_f;
};

struct NoiseFloorProfile {
    float data_f;
};

struct SideInfoDetectedPoints {
    uint16_t snr_u16;
    uint16_t noise_u16;
};

struct SphericalCoordinates {
    float range_f;
    float azimuth_f;
    float elevation_f;
    float doppler_f;
};

struct PresenceDetection {
    uint16_t detected_u16;
};

#endif //SENSORDATASTR_H