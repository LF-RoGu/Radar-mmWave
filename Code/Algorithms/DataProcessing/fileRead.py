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
    frames_data = process_log_file(file_path)
    # Count total rows in the file (excluding header)
    total_rows = sum(1 for _ in open(file_path)) - 1
    # Print summary
    print(f"\nParsed {len(frames_data)} frames successfully out of {total_rows} total rows.")

    # Print sample data (first 5 frames) with limited decimal points
    for frame_num, frame_content in list(frames_data.items())[:50]:
        print(f"\nFrame {frame_num}:")
        for tlv in frame_content["TLVs"]:
            for key, value in tlv.items():
                if isinstance(value, list):  # For lists, print each item on a new line
                    print(f"  {key}:")
                    for item in value:
                        if isinstance(item, dict):  # If the item is a dictionary, limit decimals
                            formatted_item = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in item.items()}
                            print(f"    {formatted_item}")
                        else:
                            print(f"    {item}")
                else:  # For single key-value pairs, print inline
                    print(f"  {key}: {round(value, 3) if isinstance(value, float) else value}")


