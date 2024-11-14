#ifndef TLVFRAME_H
#define TLVFRAME_H

#include "lib/SensorDataStr.h"  // For FrameHeader, TlvHeader, and DetectedObject
#include <errno.h> // Error number definitions

// RadarFrame class to manage parsing of radar data frames
class TLVheader {
public:
    //Contructor
    
    //Attributes
    FrameHeader frameHeader;                  // Holds the frame header information
    std::vector<TLVHeader> tlvHeaders;        // List of TLV headers in the frame

    //Parts

    //Setters & Getters

    //Methods
    /**
     * Function: toLittleEndian32
     * Description: Converts 4 bytes in little-endian order to a 32-bit integer.
     * Input:
     *  - data: pointer to the 4-byte array in little-endian format
     * Output:
     *  - uint32_t: the converted 32-bit integer in the correct order
     */
    uint32_t toLittleEndian32(const uint8_t* data);

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
    bool parseFrameHeader(const uint8_t* data, size_t& offset);

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
    bool parseTLVs(const uint8_t* data, size_t& offset);

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
    bool parseDetectedObjects(const uint8_t* data, size_t& offset, uint32_t length);

};

#endif // TLVFRAME_H
