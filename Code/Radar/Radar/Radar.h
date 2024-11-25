#ifndef RADAR_H
#define RADAR_H

//Global includes
#include <cstdint>
#include <vector>
#include <string>
#include <iostream>
#include <pthread.h>

//Private includes
#include "IWR6843.h"
#include "SensorData.h"

//Namespaces
using namespace std;

//External variables
extern IWR6843 sensor;

extern const int NUM_THREADS;
extern pthread_t threads[];

//Function prototypes
void* sensor_thread(void* arg);
void* controller_thread(void* arg);
void* actuator_thread(void* arg);

#endif // RADAR_H
