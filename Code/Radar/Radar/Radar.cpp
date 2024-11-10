#include "Radar.h"
#include "App.h"

std::vector<uint8_t> values;

int main() {
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
}
