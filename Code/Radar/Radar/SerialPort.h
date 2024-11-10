#ifndef SERIALPORT_H
#define SERIALPORT_H

#include "Radar.h"
#include <cerrno>
#include <cstring>
#include <fstream>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

/**
 * Function: openSerialPort
 * Description: Opens a serial port with the specified port name and baud rate.
 * Input:
 *  - portName: the name of the port to open (e.g., "/dev/ttyUSB0")
 *  - baudRate: the baud rate for the serial port (e.g., B115200)
 * Output:
 *  - int: returns the file descriptor for the opened port, or -1 on failure
 */
int openSerialPort(const char* portName, int baudRate);

/**
 * Function: configurePort
 * Description: Configures the given serial port with standard settings for communication.
 * Input:
 *  - fd: file descriptor for the open serial port
 *  - baudRate: baud rate to configure the port with
 * Output: None
 */
void configurePort(int fd, int baudRate);

/**
 * Function: readConfigFileAndSend
 * Description: Reads commands from a configuration file and sends each command to the specified serial port.
 * Input:
 *  - configPortFd: file descriptor for the configuration port
 *  - filePath: path to the configuration file containing the commands
 * Output: None
 */
void readConfigFileAndSend(int configPortFd, const std::string& filePath);

#endif // SERIALPORT_H
