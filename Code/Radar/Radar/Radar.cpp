// Radar.cpp: Definiert den Einstiegspunkt für die Anwendung.
//

#include "Radar.h"
#include "radar_sensor/IWR6843.h"

using namespace std;


#include <stdio.h>
#include <string.h>
#include <fcntl.h> // File control definitions
#include <termios.h> // POSIX terminal control definitions
#include <unistd.h> // UNIX standard function definitions
#include <errno.h> // Error number definitions
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <stdint.h>


// Constants
const int numBytes = 100 * 40;
std::vector<uint8_t> values;

// Functions for serial configuration and handling
int openSerialPort(const char* portName, int baudRate);
void configurePort(int fd, int baudRate);
void readConfigFileAndSend(int configPortFd, const std::string& filePath);
std::vector<size_t> findPatternIndexes(const std::vector<uint8_t>& values, const std::vector<uint8_t>& pattern);
std::vector<std::vector<uint8_t>> splitByPatternIndexes(const std::vector<uint8_t>& values, const std::vector<size_t>& indexes);
void readData(int dataPortFd);

int main() {
    IWR6843 sensor = IWR6843();
    sensor.init("/dev/ttyUSB0", "/dev/ttyUSB1", "../configs/xwr68xx_AOP_profile_2024_10_31T16_15_25_003.cfg");

    while (true)
    {
        sensor.poll();
    }
    return 0;
}

int openSerialPort(const char* portName, int baudRate) {
    int fd = open(portName, O_RDWR | O_NOCTTY | O_SYNC);
    if (fd < 0) {
        std::cerr << "Error opening " << portName << ": " << strerror(errno) << std::endl;
        return -1;
    }
    configurePort(fd, baudRate);
    return fd;
}

void configurePort(int fd, int baudRate) {
    struct termios tty;
    memset(&tty, 0, sizeof tty);

    if (tcgetattr(fd, &tty) != 0) {
        std::cerr << "Error getting tty attributes: " << strerror(errno) << std::endl;
        return;
    }

    cfsetospeed(&tty, baudRate);
    cfsetispeed(&tty, baudRate);

    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;     // 8-bit chars
    tty.c_iflag &= ~IGNBRK;                          // disable break processing
    tty.c_lflag = 0;                                 // no signaling chars, no echo
    tty.c_oflag = 0;                                 // no remapping, no delays
    tty.c_cc[VMIN] = 1;                             // read blocks until at least 1 char is available
    tty.c_cc[VTIME] = 5;                             // timeout in deciseconds for non-canonical read

    tty.c_iflag &= ~(IXON | IXOFF | IXANY);          // turn off s/w flow control
    tty.c_cflag |= (CLOCAL | CREAD);                 // ignore modem controls, enable reading
    tty.c_cflag &= ~(PARENB | PARODD);               // shut off parity
    tty.c_cflag &= ~CSTOPB;                          // 1 stop bit
    tty.c_cflag &= ~CRTSCTS;                         // no hardware flow control

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        std::cerr << "Error setting tty attributes: " << strerror(errno) << std::endl;
    }
}

void readConfigFileAndSend(int configPortFd, const std::string& filePath) {
    std::ifstream configFile(filePath);
    if (!configFile) {
        std::cerr << "Error opening config file: " << filePath << std::endl;
        return;
    }

    std::string line;
    while (std::getline(configFile, line)) {
        if (line.empty() || line[0] == '%') continue;

        line += "\r\n";  // Add newline character
        write(configPortFd, line.c_str(), line.size());

        // Read the response until "Done" is found
        std::string response;
        char c;
        while (read(configPortFd, &c, 1) > 0) {
            if (c == '\n' || c == '\r') {
                if (response.find("Done") != std::string::npos) {
                    std::cout << response << std::endl;
                    break;
                }
                if (!response.empty()) {
                    std::cout << response << std::endl;
                }
                response.clear();
            }
            else {
                response += c;
            }
        }
    }
}

void readData(int dataPortFd) {
    uint8_t buffer[40];
    ssize_t bytesRead = read(dataPortFd, buffer, sizeof(buffer));
    if (bytesRead > 0) {
        values.insert(values.end(), buffer, buffer + bytesRead);
    }
}

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
