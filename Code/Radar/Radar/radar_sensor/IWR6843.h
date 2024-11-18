#ifndef IWR6843_H
#define IWR6843_H

#pragma once
#include <stdio.h>
#include <string.h>
#include <fcntl.h> // File control definitions
#include <termios.h> // POSIX terminal control definitions
#include <unistd.h> // UNIX standard function definitions
#include <errno.h> // Error number definitions
#include <iostream>
#include <fstream>
#include <vector>
#include <stdint.h>
#include <sys/ioctl.h>
#include <algorithm>
#include "sensor_data/UARTFrame.h"

using namespace std;

#define DEBUG

// Add this macro for debug printing
#ifdef DEBUG
#define DEBUG_PRINT(x) std::cout << x << std::endl
#else
#define DEBUG_PRINT(x) // Do nothing
#endif

class IWR6843
{
private:
	vector<uint8_t> dataBuffer;
	int configPort_fd;
	int dataPort_fd;

	int configSerialPort(int port_fd, int baudRate);
	int sendConfigFile(int port_fd, string configFilePath);
	vector<size_t> findIndexesOfMagicWord();
	vector<vector<uint8_t>> splitIntoSublistsByIndexes(const vector<size_t>& indexes);

public:
	IWR6843();
	int init(string configPort, string dataPort, string configFilePath);
	int poll();
};

#endif // !IWR6843_H