// EndianUtils.h
#ifndef ENDIANUTILS_H
#define ENDIANUTILS_H

#include <cstdint>
#include <cstring>

class EndianUtils {
public:
    static uint16_t readLittleEndian16(const uint8_t* data, size_t& offset);
    static uint32_t readLittleEndian32(const uint8_t* data, size_t& offset);
    static int16_t readLittleEndianInt16(const uint8_t* data, size_t& offset);
    static float readFloatFromLittleEndian(const uint8_t* data, size_t& offset);
};

#endif // ENDIANUTILS_H