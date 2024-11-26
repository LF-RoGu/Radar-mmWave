#include "IWR6843.h"
#include <iomanip>  // Needed for std::setprecision
#include <fstream>
#include <filesystem> // For directory creation (C++17 and above)
#include <iostream>


IWR6843::IWR6843()
{

}

int IWR6843::init(string configPort, string dataPort, string configFilePath)
{
	configPort_fd = open(configPort.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
	if (configPort_fd < 1)
	{
		return -1;
	}
	if (configSerialPort(configPort_fd, B115200) < 1)
	{
		return -1;
	}

	dataPort_fd = open(dataPort.c_str(), O_RDONLY | O_NOCTTY | O_SYNC);
	if (dataPort_fd < 1)
	{
		return -1;
	}
	if (configSerialPort(dataPort_fd, B921600) < 1)
	{
		return -1;
	}

	if (sendConfigFile(configPort_fd, configFilePath) < 1)
	{
		return -1;
	}

	return 1;
}

int IWR6843::poll()
{
	/*
	Test for decoding
	TODO: remove when need it
	*/
	size_t offset = 0;

	//Checking if bytes are available
	int bytesAvailable = 0;
	if (ioctl(dataPort_fd, FIONREAD, &bytesAvailable) == -1)
	{
		return -1;
	}

	//Returning 0 if there are no bytes available
	if (bytesAvailable < 1)
	{
		return 0;
	}

	//Creating a temporary buffer and determining whether to read in the whole buffer or what is available (preventing overflow)
	uint8_t buffer[1024];
	int bytesToRead = min(bytesAvailable, (int)sizeof(buffer));

	//Reading bytes
	int bytesRead = read(dataPort_fd, buffer, bytesToRead);

	//Appending the read bytes to the end of the vector
	dataBuffer.insert(dataBuffer.end(), buffer, buffer + bytesRead);

	//Finding the indexes of the magic words (starting points of frames) in the buffer
	vector<size_t> indexesOfMagicWords = findIndexesOfMagicWord();

	//Returning 0 if size is below 2 (no full frame available)
	if (indexesOfMagicWords.size() < 2)
	{
		return 0;
	}

	//Deleting beginning of data until magic word if first index is unequal to 0 (garbage Data)
	if (indexesOfMagicWords.at(0) != 0)
	{
		dataBuffer.erase(dataBuffer.begin(), dataBuffer.begin() + indexesOfMagicWords.at(0));
	}

	//Extracting sublists containing one frame
	vector<vector<uint8_t>> sublists = splitIntoSublistsByIndexes(indexesOfMagicWords);

	/*
	
		ToDo: Add elements to vector of decoded items
	
	*/
	// Step 1: Parse Frame Header
	Frame_header frameHeader(sublists[0]);
	// Retrieve frame header values using getters (for debugging or further processing)
	uint32_t version = frameHeader.getVersion();
	uint32_t packetLength = frameHeader.getPacketLength();
	uint32_t platform = frameHeader.getPlatform();
	uint32_t frameNumber = frameHeader.getFrameNumber();
	uint32_t timestamp = frameHeader.getTime();
	uint32_t numObjectsDetected = frameHeader.getNumObjDetecter();
	uint32_t numTLVs = frameHeader.getNumTLV();
	uint32_t subframeNumber = frameHeader.getSubframeNum();
	// Step 2: Parse TLV Frame
	/*
	2.1. The header is parsed
	2.2. The payload is parsed and then uploaded into a struct that is a vector of data of all the possible payloads
	*/
	/*
	TODO: NOT all data types are added at this point, due to the lacking of info of what is FFT in this context, and how to get it.
	*/
	TLV_payload payloadTLV(sublists[0], numObjectsDetected);

	TLVPayloadData TLV_payload_temp = payloadTLV.getTLVFramePayloadData();

	// Print values of DetectedPoints_str if DEBUG is enabled
#ifdef DEBUG_IWR
	if (!TLV_payload_temp.DetectedPoints_str.empty()) {
		for (size_t i = 0; i < TLV_payload_temp.DetectedPoints_str.size(); ++i) {
			std::cout << std::fixed << std::setprecision(6);
			DEBUG_PRINT("!Detected Points: " << numObjectsDetected);
			const DetectedPoints& point = TLV_payload_temp.DetectedPoints_str[i];
			DEBUG_PRINT("Point... " << i + 1 << ":");
			DEBUG_PRINT("  x = " << point.x_f);
			DEBUG_PRINT("  y = " << point.y_f);
			DEBUG_PRINT("  z = " << point.z_f);
			DEBUG_PRINT("  doppler = " << point.doppler_f);
		}
	}
	else {
		DEBUG_PRINT("No detected points available.");
	}
#endif // DEBUG

#ifdef DEBUG_IWR_TXT
	if (!TLV_payload_temp.DetectedPoints_str.empty()) {
		// Define the relative path
		std::string outputPath = "../Radar/OutputFile/";

		// Ensure the directory exists
		std::filesystem::create_directories(outputPath);

		// Define the full file path
		std::string outputFile = outputPath + "detected_points.csv";

		// Open the file
		std::ofstream outFile(outputFile);

		if (!outFile.is_open()) {
			std::cerr << "Failed to create the file: " << outputFile << std::endl;
			return -1;
		}
		else
		{
			/*
			Do Nothing
			*/
		}

		outFile << "x,y,z,doppler\n"; // CSV header

		for (size_t i = 0; i < TLV_payload_temp.DetectedPoints_str.size(); ++i) {
			const DetectedPoints& point = TLV_payload_temp.DetectedPoints_str[i];
			outFile << point.x_f << "," << point.y_f << "," << point.z_f << "," << point.doppler_f << "\n";
		}

		outFile.close();
		std::cout << "Detected points exported to detected_points.csv" << std::endl;
	}
	else {
		std::cout << "No detected points available." << std::endl;
	}
#endif // DEBUG

	//Removing the elements of the dataBuffer that were processed
	dataBuffer.erase(dataBuffer.begin() + indexesOfMagicWords.front(), dataBuffer.begin() + indexesOfMagicWords.back());
	return 0;
}

int IWR6843::configSerialPort(int port_fd, int baudRate)
{
	struct termios tty;
	memset(&tty, 0, sizeof(tty));

	if (tcgetattr(port_fd, &tty) != 0)
	{
		return -1;
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

	if (tcsetattr(port_fd, TCSANOW, &tty) != 0)
	{
		return -1;
	}

	return 1;
}

int IWR6843::sendConfigFile(int port_fd, string configFilePath)
{
	ifstream configFileStream(configFilePath);
	if (!configFileStream)
	{
		return -1;
	}
	
	string configFileLine;
	while (getline(configFileStream, configFileLine))
	{
		if (configFileLine.empty() || configFileLine[0] == '%')
		{
			continue;
		}

		//configFileLine += "\n\r";
		write(port_fd, configFileLine.c_str(), configFileLine.size());

		string response;
		do
		{
			//Checking the response's content
			if (response.find("Done") != string::npos || response.find("Skipped") != string::npos)
			{
				break;
			}

			//Checking if bytes are available te read from the serial port
			int bytesAvailable = 0;
			if (ioctl(port_fd, FIONREAD, &bytesAvailable) == -1)
			{
				return -1;
			}

			//Continuing if no bytes are available
			if (bytesAvailable == 0)
			{
				continue;
			}

			//Creating a temporary buffer and determining whether to read in the whole buffer or what is available (preventing overflow)
			char buffer[1024];
			int bytesToRead = min(bytesAvailable, (int)sizeof(buffer));

			//Reading the data in
			int bytesRead = read(port_fd, buffer, bytesToRead);
			response.append(buffer, bytesRead);
		} while (true);
	}

	return 1;
}


vector<size_t> IWR6843::findIndexesOfMagicWord()
{
	const vector<uint8_t> pattern = { 0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07 };
	
	std::vector<size_t> indexes;
	auto it = dataBuffer.begin();
	while ((it = search(it, dataBuffer.end(), pattern.begin(), pattern.end())) != dataBuffer.end())
	{
		indexes.push_back(distance(dataBuffer.begin(), it));
		++it;
	}

	return indexes;
}


vector<vector<uint8_t>> IWR6843::splitIntoSublistsByIndexes(const vector<size_t>& indexes)
{
	//Preparing a return vector
	vector<vector<uint8_t>> sublists;

	//Looping through all but the last index to form sublists between consecutive indexes
	for (size_t i = 0; i < indexes.size() - 1; ++i) {
		size_t start = indexes[i];
		size_t end = indexes[i + 1];

		//Creating a sublist from dataBuffer[start] to dataBuffer[end-1] and pushing it into return vector
		vector<uint8_t> sublist(dataBuffer.begin() + start, dataBuffer.begin() + end - 1);
		sublists.push_back(sublist);
	}

	return sublists;
}