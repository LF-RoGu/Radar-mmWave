#ifndef DECODE_H
#define DECODE_H

#include "Radar.h"

std::vector<size_t> findPatternIndexes(const std::vector<uint8_t>& values, const std::vector<uint8_t>& pattern);
std::vector<std::vector<uint8_t>> splitByPatternIndexes(const std::vector<uint8_t>& values, const std::vector<size_t>& indexes);

#endif // DECODE_H
