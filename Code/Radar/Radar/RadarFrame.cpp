#include "RadarFrame.h"

// Define the expected magic word for frame synchronization
#define MAGIC_WORD 0x01020304  // Replace with the actual value from documentation

// Parses the frame header from raw data
bool RadarFrame::parseFrameHeader(const uint8_t* data, size_t& offset) {
    // Check if there's enough data for the frame header
    if (offset + sizeof(FrameHeader) > sizeof(data)) return false;

    // Extract and validate the magic word (syncWord)
    frameHeader = *reinterpret_cast<const FrameHeader*>(data + offset);
    if (frameHeader.syncWord != MAGIC_WORD) {
        std::cerr << "Invalid magic word. Expected: " << MAGIC_WORD << ", got: " << frameHeader.syncWord << "\n";
        return false;  // Invalid frame start
    }

    // Move the offset past the frame header
    offset += sizeof(FrameHeader);
    return true;
}

// Parses TLVs and populates the TLV headers and detected objects
bool RadarFrame::parseTLVs(const uint8_t* data, size_t& offset) {
    // Iterate through each TLV based on the frame header's numTLVs field
    for (size_t i = 0; i < frameHeader.numTLVs; ++i) {
        // Check if there's enough data for the TLV header
        if (offset + sizeof(TlvHeader) > sizeof(data)) return false;

        // Read the TLV header
        TlvHeader tlvHeader = *reinterpret_cast<const TlvHeader*>(data + offset);
        tlvHeaders.push_back(tlvHeader);
        offset += sizeof(TlvHeader);

        // Parse the TLV payload based on the TLV type
        if (tlvHeader.type == 1) {
            if (!parseDetectedObjects(data, offset, tlvHeader.length)) {
                std::cerr << "Error parsing detected objects\n";
                return false;
            }
        }
        else {
            // For unknown TLV types, skip the payload
            offset += tlvHeader.length;
        }
    }
    return true;
}

// Parses detected objects from the TLV payload
bool RadarFrame::parseDetectedObjects(const uint8_t* data, size_t& offset, uint32_t length) {
    // Determine the number of detected objects based on the payload length
    size_t numObjects = length / sizeof(DetectedObject);
    for (size_t i = 0; i < numObjects; ++i) {
        if (offset + sizeof(DetectedObject) > sizeof(data)) return false;

        // Read each detected object and add it to the detectedObjects list
        DetectedObject obj = *reinterpret_cast<const DetectedObject*>(data + offset);
        detectedObjects.push_back(obj);
        offset += sizeof(DetectedObject);
    }
    return true;
}