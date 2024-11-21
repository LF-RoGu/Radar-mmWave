#include "SensorData.h"


SensorData::SensorData()
{
}

SensorData::SensorData(vector<uint8_t> rawData)
{
	header = Frame_header(rawData);
	payload = TLV_payload(rawData, header.getNumTLV());
	payload_data = payload.getTLVFramePayloadData();
}

Frame_header SensorData::getHeader()
{
	return header;
}

TLV_payload SensorData::getTLVPayload()
{
	return payload;
}

TLVPayloadData SensorData::getTLVPayloadData()
{
	return payload_data;
}
