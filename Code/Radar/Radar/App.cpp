#include "App.h"

int runRadarApplication() {
    int configPortFd = openSerialPort("/dev/ttyUSB0", B115200);
    if (configPortFd < 0) return -1;

    int dataPortFd = openSerialPort("/dev/ttyUSB1", B921600);
    if (dataPortFd < 0) {
        close(configPortFd);
        return -1;
    }

    readConfigFileAndSend(configPortFd, "../configs/xwr68xx_AOP_profile_2024_10_31T16_15_25_003.cfg");

    RadarFrame radarFrame;
    size_t offset = 0;

    // Assuming you have a data buffer filled with the raw data
    uint8_t rawData[1024];

    // Attempt to parse a frame from the data
    if (radarFrame.parseFrameHeader(rawData, offset)) {
        if (radarFrame.parseTLVs(rawData, offset)) {
            std::cout << "Number of Detected Objects: " << radarFrame.detectedObjects.size() << "\n";
        }
        else {
            std::cerr << "Error parsing TLVs.\n";
            return -1;
        }
    }
    else {
        std::cerr << "Error parsing frame header.\n";
        return -1;
    }

    close(dataPortFd);
    close(configPortFd);
    return 0;
}