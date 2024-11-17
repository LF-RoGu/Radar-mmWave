// EndianUtils.cpp
#include "EndianUtils.h"
#include <stdexcept>

uint32_t EndianUtils::toLittleEndian32(std::vector<uint8_t>& data, uint8_t bytesToCheck)
{
    // Ensure the vector has at least the required number of bytes
    if (data.size() < bytesToCheck || bytesToCheck > 4) {
        throw std::invalid_argument("Invalid number of bytes to process");
    }

    uint32_t result = 0;

    // Extract little-endian 32-bit value from the vector
    for (uint8_t i = 0; i < bytesToCheck; ++i) {
        result |= static_cast<uint32_t>(data[i]) << (8 * i);
    }

    // Remove the used bytes from the beginning of the vector
    data.erase(data.begin(), data.begin() + bytesToCheck);

    return result;
}

uint64_t EndianUtils::toLittleEndian64(std::vector<uint8_t>& data, uint8_t bytesToCheck)
{
    // Ensure the vector has at least the required number of bytes
    if (data.size() < bytesToCheck || bytesToCheck > 8) {
        throw std::invalid_argument("Invalid number of bytes to process");
    }

    uint64_t result = 0;

    // Extract little-endian 64-bit value from the vector
    for (uint8_t i = 0; i < bytesToCheck; ++i) {
        result |= static_cast<uint64_t>(data[i]) << (8 * i);
    }

    // Remove the used bytes from the beginning of the vector
    data.erase(data.begin(), data.begin() + bytesToCheck);

    return result;
}
