#ifndef DECODE_H
#define DECODE_H

#include "Radar.h"

/**
 * Function: findPatternIndexes
 * Description: Searches for the specified pattern in a vector of values and returns the start indexes of matches.
 * Input:
 *  - values: vector of bytes to search within
 *  - pattern: vector of bytes representing the pattern to find
 * Output:
 *  - std::vector<size_t>: vector containing start indexes of each occurrence of the pattern
 */
std::vector<size_t> findPatternIndexes(const std::vector<uint8_t>& values, const std::vector<uint8_t>& pattern);

/**
 * Function: splitByPatternIndexes
 * Description: Splits a vector of values into sublists based on pattern indexes.
 * Input:
 *  - values: vector of bytes to be split
 *  - indexes: vector of starting indexes for each pattern match
 * Output:
 *  - std::vector<std::vector<uint8_t>>: vector of sublists divided by pattern matches
 */
std::vector<std::vector<uint8_t>> splitByPatternIndexes(const std::vector<uint8_t>& values, const std::vector<size_t>& indexes);

#endif // DECODE_H
