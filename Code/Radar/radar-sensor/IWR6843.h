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
#include <string>
#include <algorithm>
#include <stdint.h>

using namespace std;


class IWR6843
{
private:
	vector<uint8_t> dataBuffer;
	int fd_configPort;
	int fd_dataPort;
public:
	IWR6843();
	int init(string configPort, string dataPort, string configFilePath);
};

#endif // !IWR6843_H