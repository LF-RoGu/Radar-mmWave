#include "UARTFrame.h"

// Define the expected magic word in little-endian format (example value)
const uint16_t MAGIC_WORD[4] = { 0x0102, 0x0304, 0x0506, 0x0708 };

// Constructor
UARTframe::UARTframe(const std::vector<uint8_t>& uartFrame) {
    setUARTFrame(uartFrame);
    FrameHeaderData headerData = parseFrameHeader(getUARTFrame());
    // Create a FrameHeader instance using the parsed headerData
}

void UARTframe::setUARTFrame(const std::vector<uint8_t>& uartFrame) {
    UARTFrame_temp = uartFrame;
}

const std::vector<uint8_t>& UARTframe::getUARTFrame() const {
    return UARTFrame_temp;
}

// Converts up to 4 bytes in little-endian order to a 32-bit integer
uint32_t UARTframe::toLittleEndian32(const uint8_t* data, uint8_t size) {
    uint32_t result = 0;
    for (uint8_t i = 0; i < size && i < 4; ++i) {
        result |= static_cast<uint32_t>(data[i]) << (8 * i);
    }
    return result;
}

// Converts up to 8 bytes in little-endian order to a 64-bit integer
uint64_t UARTframe::toLittleEndian64(const uint8_t* data, uint8_t size) {
    uint64_t result = 0;
    for (uint8_t i = 0; i < size && i < 8; ++i) {
        result |= static_cast<uint64_t>(data[i]) << (8 * i);
    }
    return result;
}

// Function to parse the Frame Header from the raw data
FrameHeaderData UARTframe::parseFrameHeader(const std::vector<uint8_t>& data) {
    FrameHeaderData headerData;
    size_t offset = 0;

    // Parse each field with the correct byte size and update offset
    uint64_t magicWord = toLittleEndian64(&data[offset], 8);    // 8 bytes
    for (int i = 0; i < 4; ++i) {
        headerData.magicWord_u16[i] = (magicWord >> (16 * i)) & 0xFFFF;  // Extract each 2-byte segment
    }
    offset += 8;
    headerData.version_u32 = toLittleEndian32(&data[offset], 4);    // 4 bytes
    offset += 4;
    headerData.totalPacketLength_u32 = toLittleEndian32(&data[offset], 4); // 4 bytes
    offset += 4;
    headerData.platform_u32 = toLittleEndian32(&data[offset], 4);   // 4 bytes
    offset += 4;
    headerData.frameNumber_u32 = toLittleEndian32(&data[offset], 4); // 4 bytes
    offset += 4;
    headerData.timeCpuCycles_u32 = toLittleEndian32(&data[offset], 4); // 4 bytes
    offset += 4;
    headerData.numDetectedObj_u32 = toLittleEndian32(&data[offset], 4); // 4 bytes
    offset += 4;
    headerData.numTLVs_u32 = toLittleEndian32(&data[offset], 4);    // 4 bytes
    offset += 4;
    headerData.subFrameNumber_u32 = toLittleEndian32(&data[offset], 4); // 4 bytes

    return headerData;
}

UARTframe::Frame_header::Frame_header(FrameHeaderData uartFrame) {
    // Set each field using the setters
    setVersion(uartFrame.version_u32);
    setPacketLength(uartFrame.totalPacketLength_u32);
    setPlatform(uartFrame.platform_u32);
    setFrameNumber(uartFrame.frameNumber_u32);
    setTime(uartFrame.timeCpuCycles_u32);
    setNumObjDetecter(uartFrame.numDetectedObj_u32);
    setNumTLV(uartFrame.numTLVs_u32);
    setSubframeNum(uartFrame.subFrameNumber_u32);
}

bool UARTframe::Frame_header::magicWordCheck() const {
    for (int i = 0; i < 4; ++i) {
        if (FrameHeader_str.magicWord_u16[i] != MAGIC_WORD[i]) {
            return false;  // Return false if any part of the magic word doesn't match
        }
    }
    return true;  // Return true if all parts of the magic word match
}

void UARTframe::Frame_header::setVersion(uint32_t var) {
    FrameHeader_str.version_u32 = var;
}

void UARTframe::Frame_header::setPacketLength(uint32_t var) {
    FrameHeader_str.totalPacketLength_u32 = var;
}

void UARTframe::Frame_header::setPlatform(uint32_t var) {
    FrameHeader_str.platform_u32 = var;
}

void UARTframe::Frame_header::setFrameNumber(uint32_t var) {
    FrameHeader_str.frameNumber_u32 = var;
}

void UARTframe::Frame_header::setTime(uint32_t var) {
    FrameHeader_str.timeCpuCycles_u32 = var;
}

void UARTframe::Frame_header::setNumObjDetecter(uint32_t var) {
    FrameHeader_str.numDetectedObj_u32 = var;
}

void UARTframe::Frame_header::setNumTLV(uint32_t var) {
    FrameHeader_str.numTLVs_u32 = var;
}

void UARTframe::Frame_header::setSubframeNum(uint32_t var) {
    FrameHeader_str.subFrameNumber_u32 = var;
}

void UARTframe::Frame_header::setFrames()
{
}

uint32_t UARTframe::Frame_header::getVersion() {
    return FrameHeader_str.version_u32;
}

uint32_t UARTframe::Frame_header::getPacketLength() {
    return FrameHeader_str.totalPacketLength_u32;
}

uint32_t UARTframe::Frame_header::getPlatform() {
    return FrameHeader_str.platform_u32;
}

uint32_t UARTframe::Frame_header::getFrameNumber() {
    return FrameHeader_str.frameNumber_u32;
}

uint32_t UARTframe::FrameHeader::getTime() {
    return FrameHeader_str.timeCpuCycles_u32;
}

uint32_t UARTframe::FrameHeader::getNumObjDetecter() {
    return FrameHeader_str.numDetectedObj_u32;
}

uint32_t UARTframe::FrameHeader::getNumTLV() {
    return FrameHeader_str.numTLVs_u32;
}

uint32_t UARTframe::FrameHeader::getSubframeNum() {
    return FrameHeader_str.subFrameNumber_u32;
}

FrameHeaderData UARTframe::FrameHeader::getFrames()
{
    return FrameHeaderData();
}
