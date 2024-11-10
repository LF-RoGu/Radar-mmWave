#include "RadarFrame.h"

// Parses the frame header from raw data
bool RadarFrame::parseFrameHeader(const uint8_t* data, size_t& offset) {
    if (offset + sizeof(FrameHeader) > sizeof(data)) return false;

    frameHeader = *reinterpret_cast<const FrameHeader*>(data + offset);
    offset += sizeof(FrameHeader);
    return true;
}

// Parses TLVs and populates the TLV headers and detected objects
bool RadarFrame::parseTLVs(const uint8_t* data, size_t& offset) {
    for (size_t i = 0; i < frameHeader.numTLVs; ++i) {
        if (offset + sizeof(TlvHeader) > sizeof(data)) return false;

        TlvHeader tlvHeader = *reinterpret_cast<const TlvHeader*>(data + offset);
        tlvHeaders.push_back(tlvHeader);
        offset += sizeof(TlvHeader);

        // Parse TLV payload based on type; type 1 is assumed for detected objects
        if (tlvHeader.type == 1 && parseDetectedObjects(data, offset, tlvHeader.length)) {
            continue;
        }
        else {
            offset += tlvHeader.length;  // Skip unknown TLV payloads
        }
    }
    return true;
}

// Parses detected objects from the TLV payload
bool RadarFrame::parseDetectedObjects(const uint8_t* data, size_t& offset, uint32_t length) {
    size_t numObjects = length / sizeof(DetectedObject);
    for (size_t i = 0; i < numObjects; ++i) {
        if (offset + sizeof(DetectedObject) > sizeof(data)) return false;

        DetectedObject obj = *reinterpret_cast<const DetectedObject*>(data + offset);
        detectedObjects.push_back(obj);
        offset += sizeof(DetectedObject);
    }
    return true;
}
