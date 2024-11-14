#ifndef SENSORDATA_H
#define SENSORDATA_H

#pragma once
#include <algorithm>   // For std::search
#include "lib/SensorDataStr.h"

class sensorData
{
public:
	/**
	 * Function: sensorData
	 * Description: Constructor for the sensorData class.
	 * Input: None
	 * Output: None
	 */
	sensorData();

	/**
	 * Function: findPattern
	 * Description: Searches for a specified pattern in a vector of values and returns true if the pattern is found.
	 * Input:
	 *  - values: vector of bytes to search within
	 *  - pattern: vector of bytes representing the pattern to find
	 * Output:
	 *  - bool: returns true if the pattern is found, false otherwise
	 */
	bool findPattern(std::vector<uint8_t>& values, std::vector<uint8_t>& pattern);

	/**
	 * Function: setTLV
	 * Description: 
	 * Input:
	 * Output:
	 */
	std::vector<std::vector<uint8_t>> setTLV(std::vector<uint8_t>& values);

	/**
	 * Function: splitByPatternIndexes
	 * Description: Splits a vector of values into sublists based on pattern indexes.
	 * Input:
	 *  - values: vector of bytes to be split
	 *  - indexes: vector of starting indexes for each pattern match
	 * Output:
	 *  - std::vector<std::vector<uint8_t>>: vector of sublists divided by pattern matches
	 */
	std::vector<std::vector<uint8_t>> splitByPatternIndexes(std::vector<uint8_t>& values, std::vector<size_t>& indexes);
};

#endif //SENSORDATA_H