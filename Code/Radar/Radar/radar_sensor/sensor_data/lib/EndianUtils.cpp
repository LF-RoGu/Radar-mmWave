// EndianUtils.cpp
#include "EndianUtils.h"

uint16_t EndianUtils::readLittleEndian16(const uint8_t* data, size_t& offset) {
    uint16_t value = data[offset] | (data[offset + 1] << 8);
    offset += sizeof(uint16_t);
    return value;
}

uint32_t EndianUtils::readLittleEndian32(const uint8_t* data, size_t& offset) {
    uint32_t value = data[offset] | (data[offset + 1] << 8) | (data[offset + 2] << 16) | (data[offset + 3] << 24);
    offset += sizeof(uint32_t);
    return value;
}

int16_t EndianUtils::readLittleEndianInt16(const uint8_t* data, size_t& offset) {
    int16_t value = data[offset] | (data[offset + 1] << 8);
    offset += sizeof(int16_t);
    return value;
}

float EndianUtils::readFloatFromLittleEndian(const uint8_t* data, size_t& offset) {
    union {
        uint32_t i;
        float f;
    } converter;
    uint32_t intVal = readLittleEndian32(data, offset);
    converter.i = intVal;
    return converter.f;
}
