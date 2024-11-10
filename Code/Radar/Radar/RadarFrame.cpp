#include "RadarFrame.h"
#include <iostream>

// Define the expected magic word in little-endian format
#define MAGIC_WORD 0x01020304  // Adjust if needed

/**
 * Function: toLittleEndian32
 * Description: Converts 4 bytes in little-endian order to a 32-bit integer.
 * Input:
 *  - data: pointer to the 4-byte array in little-endian format
 * Output:
 *  - uint32_t: the converted 32-bit integer in the correct order
 */
uint32_t RadarFrame::toLittleEndian32(const uint8_t* data) {
    return static_cast<uint32_t>(data[0]) |
        (static_cast<uint32_t>(data[1]) << 8) |
        (static_cast<uint32_t>(data[2]) << 16) |
        (static_cast<uint32_t>(data[3]) << 24);
}

/**
 * Function: parseFrameHeader
 * Description: Parses the entire frame header from raw data, interpreting each multi-byte field
 *              in little-endian format. The magic word is used to validate the start of the frame.
 * Input:
 *  - data: pointer to the raw data buffer containing the frame
 *  - offset: reference to the current position in the data buffer, modified as data is read
 * Output:
 *  - bool: returns true if the frame header is successfully parsed and the magic word is valid
 */
bool RadarFrame::parseFrameHeader(const uint8_t* data, size_t& offset) {
    if (offset + sizeof(FrameHeader) > sizeof(data)) return false;

    // Read and verify the magic word in little-endian format
    uint32_t receivedMagicWord = toLittleEndian32(data + offset);
    if (receivedMagicWord != MAGIC_WORD) {
        std::cerr << "Invalid magic word. Expected: " << MAGIC_WORD
            << ", got: " << receivedMagicWord << "\n";
        return false;
    }

    // Parse the rest of the frame header fields in little-endian format
    frameHeader.syncWord = receivedMagicWord;
    frameHeader.version = toLittleEndian32(data + offset + 4);
    frameHeader.totalPacketLen = toLittleEndian32(data + offset + 8);
    frameHeader.platform = toLittleEndian32(data + offset + 12);
    frameHeader.frameNumber = toLittleEndian32(data + offset + 16);
    frameHeader.timeCpuCycles = toLittleEndian32(data + offset + 20);
    frameHeader.numDetectedObj = toLittleEndian32(data + offset + 24);
    frameHeader.numTLVs = toLittleEndian32(data + offset + 28);
    frameHeader.subFrameNumber = toLittleEndian32(data + offset + 32);

    offset += sizeof(FrameHeader);  // Move offset past the frame header
    return true;
}

/**
 * Function: parseTLVs
 * Description: Parses TLV (Type-Length-Value) headers from the data buffer and processes each TLV.
 *              Interprets all TLV fields in little-endian format.
 * Input:
 *  - data: pointer to the raw data buffer containing the TLVs
 *  - offset: reference to the current position in the data buffer, modified as TLVs are read
 * Output:
 *  - bool: returns true if all TLVs are successfully parsed and processed
 */
bool RadarFrame::parseTLVs(const uint8_t* data, size_t& offset) {
    for (size_t i = 0; i < frameHeader.numTLVs; ++i) {
        if (offset + sizeof(TlvHeader) > sizeof(data)) return false;

        // Read the TLV header fields in little-endian format
        TlvHeader tlvHeader;
        tlvHeader.type = toLittleEndian32(data + offset);
        tlvHeader.length = toLittleEndian32(data + offset + 4);
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

/**
 * Function: parseDetectedObjects
 * Description: Parses detected objects from a TLV payload and stores them in detectedObjects.
 *              Interprets all detected object fields in little-endian format.
 * Input:
 *  - data: pointer to the raw data buffer containing detected object data
 *  - offset: reference to the current position in the data buffer, modified as objects are read
 *  - length: length of the detected objects payload in bytes
 * Output:
 *  - bool: returns true if detected objects are successfully parsed
 */
bool RadarFrame::parseDetectedObjects(const uint8_t* data, size_t& offset, uint32_t length) {
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
