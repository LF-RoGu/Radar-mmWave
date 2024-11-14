#include "UARTFrame.h"

// Define the expected magic word in little-endian format (example value)
#define MAGIC_WORD 0x0708050604030201

// Constructor
UARTframe::UARTframe(const std::vector<uint8_t>& tlvPayload) {
    setUARTFrame(tlvPayload);
}

void UARTframe::setUARTFrame(const std::vector<uint8_t>& constUARTFrame) {
    UARTFrame_vec = constUARTFrame;
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
    FrameHeaderData header;
    size_t offset = 0;

    // Parse each field with the correct byte size and update offset
    header.magicWord_u16 = toLittleEndian64(&data[offset], 8);  // 8 bytes
    offset += 8;
    header.version_u32 = toLittleEndian32(&data[offset], 4);    // 4 bytes
    offset += 4;
    header.totalPacketLength_u32 = toLittleEndian32(&data[offset], 4); // 4 bytes
    offset += 4;
    header.platform_u32 = toLittleEndian32(&data[offset], 4);   // 4 bytes
    offset += 4;
    header.frameNumber_u32 = toLittleEndian32(&data[offset], 4); // 4 bytes
    offset += 4;
    header.timeCpuCycles_u32 = toLittleEndian32(&data[offset], 4); // 4 bytes
    offset += 4;
    header.numDetectedObj_u32 = toLittleEndian32(&data[offset], 4); // 4 bytes
    offset += 4;
    header.numTLVs_u32 = toLittleEndian32(&data[offset], 4);    // 4 bytes
    offset += 4;
    header.subFrameNumber_u32 = toLittleEndian32(&data[offset], 4); // 4 bytes

    return header;
}

bool UARTframe::FrameHeader::magicWordCheck() {
    return FrameHeader_str.magicWord_u16 == MAGIC_WORD;
}

void UARTframe::FrameHeader::setVersion(uint32_t var) {
    FrameHeader_str.version_u32 = var;
}

void UARTframe::FrameHeader::setPacketLength(uint32_t var) {
    FrameHeader_str.totalPacketLength_u32 = var;
}

void UARTframe::FrameHeader::setPlatform(uint32_t var) {
    FrameHeader_str.platform_u32 = var;
}

void UARTframe::FrameHeader::setFrameNumber(uint32_t var) {
    FrameHeader_str.frameNumber_u32 = var;
}

void UARTframe::FrameHeader::setTime(uint32_t var) {
    FrameHeader_str.timeCpuCycles_u32 = var;
}

void UARTframe::FrameHeader::setNumObjDetecter(uint32_t var) {
    FrameHeader_str.numDetectedObj_u32 = var;
}

void UARTframe::FrameHeader::setNumTLV(uint32_t var) {
    FrameHeader_str.numTLVs_u32 = var;
}

void UARTframe::FrameHeader::setSubframeNum(uint32_t var) {
    FrameHeader_str.subFrameNumber_u32 = var;
}

uint32_t UARTframe::FrameHeader::getVersion() {
    return FrameHeader_str.version_u32;
}

uint32_t UARTframe::FrameHeader::getPacketLength() {
    return FrameHeader_str.totalPacketLength_u32;
}

uint32_t UARTframe::FrameHeader::getPlatform() {
    return FrameHeader_str.platform_u32;
}

uint32_t UARTframe::FrameHeader::getFrameNumber() {
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
