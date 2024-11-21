#ifndef SENSORDATA_H
#define SENSORDATA_H

#pragma once
#include <vector>
#include <stdint.h>
#include "sensor_data/UARTFrame.h"

using namespace std;

class SensorData
{
private:
	Frame_header header;
	TLV_payload payload;
	TLVPayloadData payload_data;

public:
	SensorData();
	SensorData(vector<uint8_t> rawData);
	
	Frame_header getHeader();
	TLV_payload getTLVPayload();
	TLVPayloadData getTLVPayloadData();
};

#endif // !SENSORDATA_H