#ifndef UARTFRAME_H
#define UARTFRAME_H

#include "SensorDataStr.h"
#include <vector>
#include <cstdint>
#include <iostream>

/**
 * @class UART_frame
 * @brief Base class for handling UART frame operations such as parsing and endian conversion.
 */
class UART_frame {
protected:
    std::vector<uint8_t> UARTFrame_temp;  ///< Temporary storage for frame data.
    std::vector<uint8_t> UARTFrame_vec;  ///< Vector to hold raw UART frame data.

public:
    /**
     * Default constructor for UART_frame.
     */
    UART_frame();

    /**
     * Converts a sequence of bytes (up to 4) in little-endian order to a 32-bit integer.
     * Input:
     *  - data: Pointer to the byte array.
     *  - size: Number of bytes to convert (max 4).
     * Output:
     *  - uint32_t: The converted 32-bit integer.
     */
    static uint32_t toLittleEndian32(const uint8_t* data, uint8_t size);

    /**
     * Converts a sequence of bytes (up to 8) in little-endian order to a 64-bit integer.
     * Input:
     *  - data: Pointer to the byte array.
     *  - size: Number of bytes to convert (max 8).
     * Output:
     *  - uint64_t: The converted 64-bit integer.
     */
    static uint64_t toLittleEndian64(const uint8_t* data, uint8_t size);
};

/**
 * @class Frame_header
 * @brief Derived class for parsing and handling frame headers.
 */
class Frame_header : public UART_frame {
private:
    FrameHeaderData FrameHeader_str;  ///< Struct to store parsed frame header data.

public:

    Frame_header(const std::vector<uint8_t>& data);

    /**
     * Parses the entire frame header from raw data, interpreting each multi-byte field
     * in little-endian format. The magic word is used to validate the start of the frame.
     * Input:
     *  - data: Vector of raw data containing the frame.
     * Output:
     *  - FrameHeaderData: Parsed frame header data.
     */
    FrameHeaderData parseFrameHeader(const std::vector<uint8_t>& data);

    // Setters
    /**
     * Sets the version field in the frame header.
     * Input:
     *  - var: The version value to set.
     */
    void setVersion(uint32_t var);

    /**
     * Sets the packet length field in the frame header.
     * Input:
     *  - var: The packet length value to set.
     */
    void setPacketLength(uint32_t var);

    /**
     * Sets the platform field in the frame header.
     * Input:
     *  - var: The platform value to set.
     */
    void setPlatform(uint32_t var);

    /**
     * Sets the frame number field in the frame header.
     * Input:
     *  - var: The frame number value to set.
     */
    void setFrameNumber(uint32_t var);

    /**
     * Sets the timestamp field in the frame header.
     * Input:
     *  - var: The timestamp value to set.
     */
    void setTime(uint32_t var);

    /**
     * Sets the number of detected objects in the frame header.
     * Input:
     *  - var: The number of detected objects to set.
     */
    void setNumObjDetecter(uint32_t var);

    /**
     * Sets the number of TLVs in the frame header.
     * Input:
     *  - var: The number of TLVs to set.
     */
    void setNumTLV(uint32_t var);

    /**
     * Sets the subframe number field in the frame header.
     * Input:
     *  - var: The subframe number value to set.
     */
    void setSubframeNum(uint32_t var);

    // Getters
    /**
     * Gets the version field from the frame header.
     * Input:
     *  - None.
     * Output:
     *  - uint32_t: The version value.
     */
    uint32_t getVersion() const;

    /**
     * Gets the packet length field from the frame header.
     * Input:
     *  - None.
     * Output:
     *  - uint32_t: The packet length value.
     */
    uint32_t getPacketLength() const;

    /**
     * Gets the platform field from the frame header.
     * Input:
     *  - None.
     * Output:
     *  - uint32_t: The platform value.
     */
    uint32_t getPlatform() const;

    /**
     * Gets the frame number field from the frame header.
     * Input:
     *  - None.
     * Output:
     *  - uint32_t: The frame number value.
     */
    uint32_t getFrameNumber() const;

    /**
     * Gets the timestamp field from the frame header.
     * Input:
     *  - None.
     * Output:
     *  - uint32_t: The timestamp value.
     */
    uint32_t getTime() const;

    /**
     * Gets the number of detected objects from the frame header.
     * Input:
     *  - None.
     * Output:
     *  - uint32_t: The number of detected objects.
     */
    uint32_t getNumObjDetecter() const;

    /**
     * Gets the number of TLVs from the frame header.
     * Input:
     *  - None.
     * Output:
     *  - uint32_t: The number of TLVs.
     */
    uint32_t getNumTLV() const;

    /**
     * Gets the subframe number field from the frame header.
     * Input:
     *  - None.
     * Output:
     *  - uint32_t: The subframe number value.
     */
    uint32_t getSubframeNum() const;
};

/**
 * @class TLV_header
 * @brief Derived class for parsing TLV (Type-Length-Value) headers.
 */
class TLV_frame : public UART_frame {
public:
    /**
     * Default constructor for TLV_header.
     */
    TLV_frame();

    /**
     * Parses TLV headers from raw data and processes them.
     * Input:
     *  - data: Pointer to the raw data buffer containing TLVs.
     *  - offset: Reference to the current offset in the data buffer.
     * Output:
     *  - bool: True if all TLVs are successfully parsed and processed, false otherwise.
     */
    bool parseTLVHeader(const uint8_t* data, size_t& offset);
};

#endif // UARTFRAME_H
