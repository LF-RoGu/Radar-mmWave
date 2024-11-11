#include "IWR6843.h"

IWR6843::IWR6843()
{

}

int IWR6843::init(string configPort, string dataPort, string configFilePath)
{
	fd_configPort = open(configPort, O_RDWR, | O_NCTTY | O_SYNC;

	return 0;
}
