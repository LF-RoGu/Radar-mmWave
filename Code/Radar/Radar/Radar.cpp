#include "Radar.h"
#include "App.h"
#include "radar_sensor/IWR6843.h"

std::vector<uint8_t> values;

int main() {
    IWR6843 sensor = IWR6843();
    sensor.init("/dev/ttyUSB0", "/dev/ttyUSB1", "../configs/xwr68xx_AOP_profile_2024_10_31T16_15_25_003.cfg");

    /*
    // Variable to control the number of frames received
    int receivedFrames = 0;

    // Run the radar application in a loop
    while (CONTINUOUS_READING || receivedFrames < FIXED_FRAME_COUNT) {
        // Call the radar application function
        if (runRadarApplication() < 0) {
            std::cerr << "Error in radar application. Exiting.\n";
            break;
        }

        // Increment frame count if reading a fixed number of frames
        if (!CONTINUOUS_READING) {
            receivedFrames++;
            std::cout << "Frame " << receivedFrames << " received.\n";
        }
    }

    std::cout << "Received total frames: " << receivedFrames << "\n";
    return 0;
    */
}
