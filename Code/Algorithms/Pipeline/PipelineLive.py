import serial
import time
import threading
import struct
import queue
from threading import Lock

import dataDecoder

# -------------------------------
# Configuration Commands
# -------------------------------
CONFIG_COMMANDS = [
    "sensorStop",
    "flushCfg",
    "dfeDataOutputMode 1",
    "channelCfg 15 7 0",
    "adcCfg 2 1",
    "adcbufCfg -1 0 1 1 1",
    "profileCfg 0 60 46 7 18.24 0 0 82.237 1 128 12499 0 0 158",
    "chirpCfg 0 0 0 0 0 0 0 1",
    "chirpCfg 1 1 0 0 0 0 0 2",
    "frameCfg 0 1 128 0 33.333 1 0",
    "lowPower 0 0",
    "guiMonitor -1 1 0 0 0 0 0",
    "cfarCfg -1 0 2 8 4 3 0 15 1",
    "cfarCfg -1 1 0 8 4 4 1 15 1",
    "multiObjBeamForming -1 1 0.5",
    "clutterRemoval -1 0",
    "calibDcRangeSig -1 0 -5 8 256",
    "extendedMaxVelocity -1 0",
    "lvdsStreamCfg -1 0 0 0",
    "compRangeBiasAndRxChanPhase 0.0 1 0 -1 0 1 0 -1 0 1 0 -1 0 1 0 -1 0 1 0 -1 0 1 0 -1 0",
    "measureRangeBiasAndRxChanPhase 0 1.5 0.2",
    "CQRxSatMonitor 0 3 4 31 0",
    "CQSigImgMonitor 0 63 4",
    "analogMonitor 0 0",
    "aoaFovCfg -1 -90 90 -90 90",
    "cfarFovCfg -1 0 0 18.23",
    "cfarFovCfg -1 1 -9.72 9.72",
    "calibData 0 0 0",
    "sensorStart"
]

frame_queue = queue.Queue()

write_lock = threading.Lock()
read_lock = threading.Lock()

# -------------------------------
# Send Configuration to Sensor
# -------------------------------
def send_configuration(port='COM4', baudrate=115200):
    ser = serial.Serial(port, baudrate, timeout=1)
    time.sleep(2)

    for command in CONFIG_COMMANDS:
        ser.write((command + '\n').encode())
        print(f"Sent: {command}")
        time.sleep(0.1)
    ser.close()

# -------------------------------
# Sensor Reading Thread
# -------------------------------
def sensor_thread(port='COM8', baudrate=921600):
    ser = serial.Serial(port, baudrate, timeout=1)
    magic_word = b'\x02\x01\x04\x03\x06\x05\x08\x07'
    buffer = bytearray()

    while True:
        if ser.in_waiting:
            buffer.extend(ser.read(ser.in_waiting))

            while magic_word in buffer:
                start_idx = buffer.find(magic_word)
                end_idx = buffer.find(magic_word, start_idx + 8)

                if end_idx == -1:
                    break

                frame = buffer[start_idx:end_idx]

                # Thread-safe write operation
                with write_lock:
                    frame_queue.put(frame)

                buffer = buffer[end_idx:]


# -------------------------------
# Data Parsing Thread
# -------------------------------
def parsing_thread():
    while True:
        frame = None

        # Thread-safe read operation
        with read_lock:
            if not frame_queue.empty():
                frame = frame_queue.get()

        if frame:
            print("Frame Data:")
            print(frame)
            print("------------------")
            try:
                # Decode the frame
                decoded_frames = dataDecoder.dataToFrames(frame)

                for decoded in decoded_frames:
                    print("Decoded Frame:")
                    print(decoded)
                    print("------------------")
            except Exception as e:
                print(f"Error decoding frame: {e}")


# -------------------------------
# Start Threads
# -------------------------------
if __name__ == "__main__":
    send_configuration(port='COM4')
    
    threading.Thread(target=sensor_thread, daemon=True).start()
    threading.Thread(target=parsing_thread, daemon=True).start()

    while True:
        time.sleep(1)
