#include "ReadData.h"

void readData(int dataPortFd) {
    uint8_t buffer[40];
    ssize_t bytesRead = read(dataPortFd, buffer, sizeof(buffer));
    if (bytesRead > 0) {
        values.insert(values.end(), buffer, buffer + bytesRead);
    }
}
