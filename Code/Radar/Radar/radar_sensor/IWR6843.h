#ifndef IWR6843_H
#define IWR6843_H

#pragma once
#include <string.h>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <fstream>
#include <vector>
#include <stdint.h>
#include <sys/ioctl.h>
#include <algorithm>
#include <pthread.h>

#include "SensorData.h"

using namespace std;

class IWR6843
{
private:
	int configPort_fd;
	int dataPort_fd;

	vector<uint8_t> dataBuffer;
	vector<SensorData> decodedFrameBuffer;
	pthread_mutex_t decodedFrameBufferMutex;

	int configSerialPort(int port_fd, int baudRate);
	int sendConfigFile(int port_fd, string configFilePath);
	vector<size_t> findIndexesOfMagicWord();
	vector<vector<uint8_t>> splitIntoSublistsByIndexes(const vector<size_t>& indexes);

public:
	IWR6843();
	~IWR6843();
	int init(string configPort, string dataPort, string configFilePath);
	int poll();
	vector<SensorData> getDecodedFramesFromTop(int num, bool del);
	int getDecodedFramesSize();
};

#endif // !IWR6843_H