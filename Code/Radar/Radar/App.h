#ifndef APP_H
#define APP_H

#include "Radar.h"
#include "SerialPort.h"
#include "Decode.h"
#include "ReadData.h"
#include "RadarFrame.h"

/**
 * Function: runRadarApplication
 * Description: Manages the radar application workflow, including configuring ports, reading data,
 *              and processing frames.
 * Input: None
 * Output:
 *  - int: returns 0 on successful execution, or -1 on error
 */
int runRadarApplication();

#endif // APP_H
