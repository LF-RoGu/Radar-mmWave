// EndianUtils.h
#ifndef ENDIANUTILS_H
#define ENDIANUTILS_H

#include <cstdint>
#include <cstring>

class EndianUtils {
public:
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

#endif // ENDIANUTILS_H