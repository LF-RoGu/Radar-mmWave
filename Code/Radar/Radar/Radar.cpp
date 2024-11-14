#include "Radar.h"
#include "radar_sensor/IWR6843.h"

std::vector<uint8_t> values;

int main() {
    IWR6843 sensor = IWR6843();
    sensor.init("/dev/ttyUSB0", "/dev/ttyUSB1", "../configs/xwr68xx_AOP_profile_2024_10_31T16_15_25_003.cfg");

    while (true)
    {
        sensor.poll();
    }
    return 0;
}
