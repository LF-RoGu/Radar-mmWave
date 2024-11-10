#ifndef READDATA_H
#define READDATA_H

#include "Radar.h"
#include <unistd.h>

/**
 * Function: readData
 * Description: Reads data from the specified data port and appends it to the global values vector.
 * Input:
 *  - dataPortFd: file descriptor for the data port
 * Output: None
 */
void readData(int dataPortFd);

#endif // READDATA_H
