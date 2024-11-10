#ifndef RADARFRAME_H
#define RADARFRAME_H

#include <cstdint>
#include <vector>
#include "Radar.h"  // For FrameHeader, TlvHeader, and DetectedObject

struct RadarFrame {
    FrameHeader frameHeader;
    std::vector<TlvHeader> tlvHeaders;
    std::vector<DetectedObject> detectedObjects;

    bool parseFrameHeader(const uint8_t* data, size_t& offset);
    bool parseTLVs(const uint8_t* data, size_t& offset);
    bool parseDetectedObjects(const uint8_t* data, size_t& offset, uint32_t length);
};

#endif // RADARFRAME_H
