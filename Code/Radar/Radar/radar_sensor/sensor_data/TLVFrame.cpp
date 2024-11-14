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
    uint32_t receivedMagicWord = toLittleEndian32(data + offset);
    if (receivedMagicWord != MAGIC_WORD) {
        std::cerr << "Invalid magic word.\n";
        return false;
    }
    // Parse remaining fields...
    offset += sizeof(FrameHeader);
    return true;
}