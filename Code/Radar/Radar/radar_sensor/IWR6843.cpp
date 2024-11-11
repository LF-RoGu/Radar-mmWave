#include "IWR6843.h"

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
