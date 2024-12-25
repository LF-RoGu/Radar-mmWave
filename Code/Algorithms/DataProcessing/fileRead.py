import os
import pandas as pd
import struct
import math

from radar_utilsProcessing import *

# Main script
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = os.path.join("..", "..", "..", "Logs", "LogsPart3", "StaticMonitoring", "Test_30fps_dist15mts_vehicleLog_5mps_d3x3wall_att1_log.csv")
    file_path = os.path.normpath(os.path.join(script_dir, relative_path))

    print(f"Processing file: {file_path}")

    # Instantiate the parser and process the log file
    parser = RadarLogParser(file_path, snr_threshold=15.0)
    frames = parser.process_log_file()

    # Count total rows in the file (excluding header)
    total_rows = sum(1 for _ in open(file_path)) - 1

    # Print summary
    print(f"\nParsed {len(frames)} frames successfully out of {total_rows} total rows.")

    # Print sample data (first 5 frames) with limited decimal points
    for i, frame in enumerate(frames[:5]):
        print(f"\nFrame {frame.header.frame_number}:")

        for tlv in frame.tlvs:
            if isinstance(tlv, Type1Data):
                print("  Type 1 Data:")
                for point in tlv.coordinates:
                    print(f"    X: {round(point.x, 3)}, Y: {round(point.y, 3)}, Z: {round(point.z, 3)}, Doppler: {round(point.doppler, 3)}")
            elif isinstance(tlv, Type7Data):
                print("  Type 7 Data:")
                for info in tlv.side_info:
                    print(f"    SNR: {round(info.snr, 3)} dB, Noise: {round(info.noise, 3)} dB")
                    
