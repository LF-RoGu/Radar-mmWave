// EndianUtils.cpp
#include "EndianUtils.h"

uint32_t EndianUtils::toLittleEndian32(const uint8_t* data, uint8_t size)
{
    uint32_t result = 0;
    for (uint8_t i = 0; i < size && i < 4; ++i) {
        result |= static_cast<uint32_t>(data[i]) << (8 * i);
    }
    return result;
}

uint64_t EndianUtils::toLittleEndian64(const uint8_t* data, uint8_t size)
{
    uint64_t result = 0;
    for (uint8_t i = 0; i < size && i < 8; ++i) {
        result |= static_cast<uint64_t>(data[i]) << (8 * i);
    }
    return result;
}
