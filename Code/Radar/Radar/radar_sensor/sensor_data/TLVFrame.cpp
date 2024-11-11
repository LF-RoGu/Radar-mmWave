#include "TLVFrame.h"


// Define the expected magic word in little-endian format
#define MAGIC_WORD 0x01020304  // Adjust if needed

uint32_t TLVheader::toLittleEndian32(const uint8_t* data) {
    return static_cast<uint32_t>(data[0]) |
        (static_cast<uint32_t>(data[1]) << 8) |
        (static_cast<uint32_t>(data[2]) << 16) |
        (static_cast<uint32_t>(data[3]) << 24);
}

bool TLVheader::parseFrameHeader(const uint8_t* data, size_t& offset) {
    if (offset + sizeof(FrameHeader) > sizeof(data)) return false;

    // Read and verify the magic word in little-endian format
    uint32_t receivedMagicWord = toLittleEndian32(data + offset);
    if (receivedMagicWord != MAGIC_WORD) {
        std::cerr << "Invalid magic word. Expected: " << MAGIC_WORD
            << ", got: " << receivedMagicWord << "\n";
        return false;
    }

    // Parse the rest of the frame header fields in little-endian format
    frameHeader.frameNumber_u32 = receivedMagicWord;
    frameHeader.version_u32 = toLittleEndian32(data + offset + 4);
    frameHeader.totalPacketLength_u32 = toLittleEndian32(data + offset + 8);
    frameHeader.platform_u32 = toLittleEndian32(data + offset + 12);
    frameHeader.frameNumber_u32 = toLittleEndian32(data + offset + 16);
    frameHeader.timeCpuCycles_u32 = toLittleEndian32(data + offset + 20);
    frameHeader.numDetectedObj_u32 = toLittleEndian32(data + offset + 24);
    frameHeader.numTLVs_u32 = toLittleEndian32(data + offset + 28);
    frameHeader.subFrameNumber_u32 = toLittleEndian32(data + offset + 32);

    offset += sizeof(FrameHeader);  // Move offset past the frame header
    return true;
}

bool TLVheader::parseTLVs(const uint8_t* data, size_t& offset) {
    for (size_t i = 0; i < frameHeader.numTLVs; ++i) {
        if (offset + sizeof(TlvHeader) > sizeof(data)) return false;

        // Read the TLV header fields in little-endian format
        TlvHeader tlvHeaders;
        tlvHeaders.type = toLittleEndian32(data + offset);
        tlvHeaders.length = toLittleEndian32(data + offset + 4);
        tlvHeaders.push_back(tlvHeader);
        offset += sizeof(TlvHeader);

        // Parse the TLV payload based on type
        if (tlvHeader.type == 1) {
            if (!parseDetectedObjects(data, offset, tlvHeader.length)) {
                std::cerr << "Error parsing detected objects\n";
                return false;
            }
        }
        else {
            // Skip unknown TLV payloads
            offset += tlvHeader.length;
        }
    }
    return true;
}

bool TLVheader::parseDetectedObjects(const uint8_t* data, size_t& offset, uint32_t length) {
    size_t numObjects = length / sizeof(DetectedObject);
    for (size_t i = 0; i < numObjects; ++i) {
        if (offset + sizeof(DetectedObject) > sizeof(data)) return false;

        // Read each detected object and add it to the detectedObjects list
        DetectedObject obj = *reinterpret_cast<const DetectedObject*>(data + offset);
        obj.range = toLittleEndian32(reinterpret_cast<const uint8_t*>(&obj.range));
        obj.doppler = toLittleEndian32(reinterpret_cast<const uint8_t*>(&obj.doppler));
        obj.peakVal = toLittleEndian32(reinterpret_cast<const uint8_t*>(&obj.peakVal));
        obj.x = toLittleEndian32(reinterpret_cast<const uint8_t*>(&obj.x));
        obj.y = toLittleEndian32(reinterpret_cast<const uint8_t*>(&obj.y));
        obj.z = toLittleEndian32(reinterpret_cast<const uint8_t*>(&obj.z));
        detectedObjects.push_back(obj);
        offset += sizeof(DetectedObject);
    }
    return true;
}
