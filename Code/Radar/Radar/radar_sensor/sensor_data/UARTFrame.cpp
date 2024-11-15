#include "UARTFrame.h"

constexpr uint64_t MAGIC_WORD = 0x0708050604030201;

UART_frame::UART_frame() {}

uint32_t UART_frame::toLittleEndian32(const uint8_t* data, uint8_t size) 
{
    uint32_t result = 0;
    for (uint8_t i = 0; i < size && i < 4; ++i) {
        result |= static_cast<uint32_t>(data[i]) << (8 * i);
    }
    return result;
}

uint64_t UART_frame::toLittleEndian64(const uint8_t* data, uint8_t size) 
{
    uint64_t result = 0;
    for (uint8_t i = 0; i < size && i < 8; ++i) {
        result |= static_cast<uint64_t>(data[i]) << (8 * i);
    }
    return result;
}

Frame_header::Frame_header(const std::vector<uint8_t>& data) 
{
    parseFrameHeader(data);
}

FrameHeaderData Frame_header::parseFrameHeader(const std::vector<uint8_t>& data) 
{
    FrameHeaderData headerData;
    size_t offset = 0;

    uint64_t magicWord = toLittleEndian64(&data[offset], 8);
    // Check if the magic word matches the expected value
    if (magicWord != MAGIC_WORD) {
        std::cerr << "Error: Invalid magic word detected! Aborting frame parsing.\n";
        return {}; // Return an empty FrameHeaderData or handle error appropriately
    }
    for (int i = 0; i < 4; ++i) {
        headerData.magicWord_u16[i] = (magicWord >> (16 * i)) & 0xFFFF;
    }
    offset += 8;

    headerData.version_u32 = toLittleEndian32(&data[offset], 4);
    offset += 4;

    headerData.totalPacketLength_u32 = toLittleEndian32(&data[offset], 4);
    offset += 4;

    headerData.platform_u32 = toLittleEndian32(&data[offset], 4);
    offset += 4;

    headerData.frameNumber_u32 = toLittleEndian32(&data[offset], 4);
    offset += 4;

    headerData.timeCpuCycles_u32 = toLittleEndian32(&data[offset], 4);
    offset += 4;

    headerData.numDetectedObj_u32 = toLittleEndian32(&data[offset], 4);
    offset += 4;

    headerData.numTLVs_u32 = toLittleEndian32(&data[offset], 4);
    offset += 4;

    headerData.subFrameNumber_u32 = toLittleEndian32(&data[offset], 4);

    return headerData;
}

void Frame_header::setVersion(uint32_t var)
{
    FrameHeader_str.version_u32 = var;
}

void Frame_header::setPacketLength(uint32_t var)
{
    FrameHeader_str.totalPacketLength_u32 = var;
}

void Frame_header::setPlatform(uint32_t var)
{
    FrameHeader_str.platform_u32 = var;
}

void Frame_header::setFrameNumber(uint32_t var)
{
    FrameHeader_str.frameNumber_u32 = var;
}

void Frame_header::setTime(uint32_t var)
{
    FrameHeader_str.timeCpuCycles_u32 = var;
}

void Frame_header::setNumObjDetecter(uint32_t var)
{
    FrameHeader_str.numDetectedObj_u32 = var;
}

void Frame_header::setNumTLV(uint32_t var)
{
    FrameHeader_str.numTLVs_u32 = var;
}

void Frame_header::setSubframeNum(uint32_t var)
{
    FrameHeader_str.subFrameNumber_u32 = var;
}

uint32_t Frame_header::getVersion() const
{
    return FrameHeader_str.version_u32;
}

uint32_t Frame_header::getPacketLength() const
{
    return FrameHeader_str.totalPacketLength_u32;
}

uint32_t Frame_header::getPlatform() const
{
    return FrameHeader_str.platform_u32;
}

uint32_t Frame_header::getFrameNumber() const
{
    return FrameHeader_str.frameNumber_u32;
}

uint32_t Frame_header::getTime() const
{
    return FrameHeader_str.timeCpuCycles_u32;
}

uint32_t Frame_header::getNumObjDetecter() const
{
    return FrameHeader_str.numDetectedObj_u32;
}

uint32_t Frame_header::getNumTLV() const
{
    return FrameHeader_str.numTLVs_u32;
}

uint32_t Frame_header::getSubframeNum() const
{
    return FrameHeader_str.subFrameNumber_u32;
}


TLV_frame::TLV_frame() {}

bool TLV_frame::parseTLVHeader(const uint8_t* data, size_t& offset) {
    // TLV parsing logic should be implemented here
    return true;
}
