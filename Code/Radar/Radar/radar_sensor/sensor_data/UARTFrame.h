#ifndef UARTFRAME_H
#define UARTFRAME_H

#include "lib/SensorDataStr.h"  // For FrameHeaderData, TLVHeaderData, and DetectedPoints
#include <vector>
#include <cstdint>
#include <iostream>

class UARTframe {
private:
    std::vector<uint8_t> UARTFrame_vec;  // Raw UART frame data

public:
    // Constructor
    UARTframe(const std::vector<uint8_t>& uartFrame);

    // Sets the UART frame data
    void setUARTFrame(const std::vector<uint8_t>& uartFrame);

    const std::vector<uint8_t>& getUARTFrame() const;

    // Converts up to 4 bytes in little-endian order to a 32-bit integer
    static uint32_t toLittleEndian32(const uint8_t* data, uint8_t size);

    // Converts up to 8 bytes in little-endian order to a 64-bit integer
    static uint64_t toLittleEndian64(const uint8_t* data, uint8_t size);

    /**
     * Parses the entire frame header from raw data, interpreting each multi-byte field
     * in little-endian format. The magic word is used to validate the start of the frame.
     * Input:
     *  - data: vector of raw data containing the frame
     * Output:
     *  - FrameHeaderData: Parsed frame header data
     */
    FrameHeaderData parseFrameHeader(const std::vector<uint8_t>& data);

    /**
     * Parses TLV (Type-Length-Value) headers from the data buffer and processes each TLV.
     * Input:
     *  - data: pointer to the raw data buffer containing the TLVs
     *  - offset: reference to the current position in the data buffer, modified as TLVs are read
     * Output:
     *  - bool: returns true if all TLVs are successfully parsed and processed
     */
    bool parseTLVs(const uint8_t* data, size_t& offset);

    /**
     * Parses detected objects from a TLV payload and stores them in detectedObjects.
     * Input:
     *  - data: pointer to the raw data buffer containing detected object data
     *  - offset: reference to the current position in the data buffer, modified as objects are read
     *  - length: length of the detected objects payload in bytes
     * Output:
     *  - bool: returns true if detected objects are successfully parsed
     */
    bool parseDetectedObjects(const uint8_t* data, size_t& offset, uint32_t length);

    class FrameHeader {
    private:
        FrameHeaderData FrameHeader_str;

    public:
        bool magicWordCheck();

        void setVersion(uint32_t var);
        void setPacketLength(uint32_t var);
        void setPlatform(uint32_t var);
        void setFrameNumber(uint32_t var);
        void setTime(uint32_t var);
        void setNumObjDetecter(uint32_t var);
        void setNumTLV(uint32_t var);
        void setSubframeNum(uint32_t var);

        uint32_t getVersion();
        uint32_t getPacketLength();
        uint32_t getPlatform();
        uint32_t getFrameNumber();
        uint32_t getTime();
        uint32_t getNumObjDetecter();
        uint32_t getNumTLV();
        uint32_t getSubframeNum();
    };
};

#endif // UARTFRAME_H
