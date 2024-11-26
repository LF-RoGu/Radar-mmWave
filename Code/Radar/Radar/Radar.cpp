// Radar.cpp: Definiert den Einstiegspunkt für die Anwendung.
//

#include "Radar.h"

IWR6843 sensor;

const int NUM_THREADS = 3;
pthread_t threads[NUM_THREADS];

int main() {
    
    //Initializing the sensor
    sensor = IWR6843();
    sensor.init("/dev/ttyUSB0", "/dev/ttyUSB1", "../configs/xwr68xx_AOP_profile_2024_10_31T16_15_25_003.cfg");

    //Creating an array holding the function pointers for the threads
    void* (*thread_functions[NUM_THREADS])(void*) =
    {
        sensor_thread,
        controller_thread,
        actuator_thread
    };
    
    //Creating the threads
    for (int i = 0; i < NUM_THREADS; i++)
    {
        if (pthread_create(&threads[i], nullptr, thread_functions[i], nullptr) != 0)
        {
            cout << "Error creating thread with ID " << i + 1 << endl;
            return -1;
        }
    }

    //Joining the threads
    for (int i = 0; i < NUM_THREADS; i++)
    {
        if (pthread_join(threads[i], nullptr) != 0)
        {
            cout << "Error joining thread with ID " << i + 1 << endl;
            return -1;
        }
    }

    //Returning a 1 after joining all threads (may not be reached but for the sake of completeness)
    cout << "Successfully joined all threads" << endl;
    return 1;
}

/// <summary>
/// Function of the sensor thread
/// </summary>
/// <param name="arg"></param>
/// <returns></returns>
void* sensor_thread(void* arg)
{
    //Obtaining the thread's ID
    int thread_id = pthread_self();

    while (true)
    {
        //Polling the sensor and getting the amount of recently received frames
        int numOfNewFrames = sensor.poll();

        //Continuing if no new frames are available
        if (numOfNewFrames < 1)
        {
            continue;
        }

        //Processing if any new frames were received
        //Getting and deleting the new frames from the buffer of decoded frames
        vector<SensorData> newFrames;
        sensor.copyDecodedFramesFromTop(newFrames, numOfNewFrames, true, 100);

        //Iterating over all new frames and printing out the x,y,z,doppler values
        for (int i = 0; i < newFrames.size(); i++)
        {
            cout << "Frame " << i << endl;
            vector<DetectedPoints> points = newFrames.at(i).getTLVPayloadData().DetectedPoints_str;
            for (int n = 0; n < points.size(); n++)
            {
                cout << "Point " << n << ":" << endl;
                cout << "x: " << points.at(n).x_f << endl;
                cout << "y: " << points.at(n).y_f << endl;
                cout << "z: " << points.at(n).z_f << endl;
                cout << "doppler: " << points.at(n).doppler_f << endl;
            }
        }
    }

    //Exiting the thread
    pthread_exit(nullptr);
}

/// <summary>
/// Function of the controller thread
/// </summary>
/// <param name="arg"></param>
/// <returns></returns>
void* controller_thread(void* arg)
{
    //Obtaining the thread's ID
    int thread_id = pthread_self();

    //Simulating work
    cout << "Hello from thread " << thread_id << endl;
    while (true)
    {
        sleep(10);
    }
    
    //Exiting the thread
    pthread_exit(nullptr);
}

/// <summary>
/// Function of the actuator thread
/// </summary>
/// <param name="arg"></param>
/// <returns></returns>
void* actuator_thread(void* arg)
{
    //Obtaining the thread's ID
    int thread_id = pthread_self();
    
    //Simulating work
    cout << "Hello from thread " << thread_id << endl;
    while (true)
    {
        sleep(10);
    }
    
    //Exiting the thread
    pthread_exit(nullptr);
}