#ifndef SERIALPORT_H
#define SERIALPORT_H

#include "Radar.h"

#include <cerrno>       // Error number definitions
#include <cstring>      // String manipulation functions
#include <fstream>      // For file handling

#include <fcntl.h>      // File control definitions
#include <termios.h>    // POSIX terminal control definitions
#include <unistd.h>     // UNIX standard function definitions

int openSerialPort(const char* portName, int baudRate);
void configurePort(int fd, int baudRate);
void readConfigFileAndSend(int configPortFd, const std::string& filePath);

#endif // SERIALPORT_H
