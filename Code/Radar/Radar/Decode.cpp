#include "Decode.h"
#include <algorithm>   // For std::search

std::vector<size_t> findPatternIndexes(const std::vector<uint8_t>& values, const std::vector<uint8_t>& pattern) {
    std::vector<size_t> indexes;
    auto it = values.begin();
    while ((it = std::search(it, values.end(), pattern.begin(), pattern.end())) != values.end()) {
        indexes.push_back(std::distance(values.begin(), it));
        ++it;
    }
    return indexes;
}

std::vector<std::vector<uint8_t>> splitByPatternIndexes(const std::vector<uint8_t>& values, const std::vector<size_t>& indexes) {
    std::vector<std::vector<uint8_t>> sublists;
    for (size_t j = 0; j < indexes.size(); ++j) {
        size_t start = indexes[j];
        size_t end = (j < indexes.size() - 1) ? indexes[j + 1] : values.size();
        sublists.emplace_back(values.begin() + start, values.begin() + end);
    }
    return sublists;
}
