#ifndef RADARFRAME_H
#define RADARFRAME_H

#include <cstdint>
#include <vector>
#include "Radar.h"  // For FrameHeader, TlvHeader, and DetectedObject

struct RadarFrame {
    FrameHeader frameHeader;                  // Holds the frame header information
    std::vector<TlvHeader> tlvHeaders;        // List of TLV headers in the frame
    std::vector<DetectedObject> detectedObjects; // List of detected objects from TLVs

    /**
     * Function: parseFrameHeader
     * Description: Parses the frame header from raw data, including validating the magic word.
     * Input:
     *  - data: pointer to the raw data buffer containing the frame
     *  - offset: reference to the current position in the data buffer, modified as data is read
     * Output:
     *  - bool: returns true if the frame header is successfully parsed and the magic word is valid
     */
    bool parseFrameHeader(const uint8_t* data, size_t& offset);

    /**
     * Function: parseTLVs
     * Description: Parses TLV (Type-Length-Value) headers from the data buffer and processes each TLV.
     * Input:
     *  - data: pointer to the raw data buffer containing the TLVs
     *  - offset: reference to the current position in the data buffer, modified as TLVs are read
     * Output:
     *  - bool: returns true if all TLVs are successfully parsed and processed
     */
    bool parseTLVs(const uint8_t* data, size_t& offset);

    /**
     * Function: parseDetectedObjects
     * Description: Parses detected objects from a TLV payload and stores them in detectedObjects.
     * Input:
     *  - data: pointer to the raw data buffer containing detected object data
     *  - offset: reference to the current position in the data buffer, modified as objects are read
     *  - length: length of the detected objects payload in bytes
     * Output:
     *  - bool: returns true if detected objects are successfully parsed
     */
    bool parseDetectedObjects(const uint8_t* data, size_t& offset, uint32_t length);
};

#endif // RADARFRAME_H
