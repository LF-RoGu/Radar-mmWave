// Radar.cpp: Definiert den Einstiegspunkt für die Anwendung.
//

#include "Radar.h"
#include "IWR6843.h"
#include "SensorData.h"

using namespace std;


int main() {
    IWR6843 sensor = IWR6843();
    sensor.init("/dev/ttyUSB0", "/dev/ttyUSB1", "../configs/xwr68xx_AOP_profile_2024_10_31T16_15_25_003.cfg");

    while (true)
    {
        //Polling the sensor and getting the amount of recently received frames
        int numOfNewFrames = sensor.poll();
        
        //Processing if any new frames were received
        if (numOfNewFrames > 0)
        {
            //Getting and deleting the new frames from the buffer of decoded frames
            vector<SensorData> newFrames = sensor.getDecodedFramesFromTop(numOfNewFrames, true);
            
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
    }
    return 0;
}