#include "SerialPort.h"

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

    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_iflag &= ~IGNBRK;
    tty.c_lflag = 0;
    tty.c_oflag = 0;
    tty.c_cc[VMIN] = 1;
    tty.c_cc[VTIME] = 5;
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~(PARENB | PARODD);
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;

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

        line += "\r\n";
        write(configPortFd, line.c_str(), line.size());

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
