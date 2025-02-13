import serial
import time

__all__ = ['sendConfiguration']


def send_configuration(configuration_commands, configuration_port, baudrate=115200):
    """! Configures the IWR6843 via UART.
    @param configuration_commands   List of commands to initialize the sensor.
    @param configuration_port   COM-Port for sending the commands to the sensor.
    @param baudrate Baudrate used for communicating with the configuration COM-Port (standard 115200).
    @return 
    """
    
    ser = serial.Serial(configuration_port, baudrate, timeout=1)
    time.sleep(2)

    for command in configuration_commands:
        ser.write((command + '\n').encode())
        print(f"Sent: {command}")
        time.sleep(0.1)
    ser.close()