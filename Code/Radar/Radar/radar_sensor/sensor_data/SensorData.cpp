#include "SensorData.h"

sensorData::sensorData()
{
	//Todo: Add initialization of the sensorData
}

bool sensorData::findPattern(std::vector<uint8_t>& values, std::vector<uint8_t>& pattern) {
    // Use std::search to find the pattern in the values vector
    auto it = std::search(values.begin(), values.end(), pattern.begin(), pattern.end());

    // If the pattern is found, std::search will return an iterator to the start of the match
    // If not found, it will return values.end()
    return it != values.end();
}

std::vector<std::vector<uint8_t>> sensorData::splitByPatternIndexes(std::vector<uint8_t>& values, std::vector<size_t>& indexes) {
    std::vector<std::vector<uint8_t>> sublists;
    for (size_t j = 0; j < indexes.size(); ++j) {
        size_t start = indexes[j];
        size_t end = (j < indexes.size() - 1) ? indexes[j + 1] : values.size();
        sublists.emplace_back(values.begin() + start, values.begin() + end);
    }
    return sublists;
}