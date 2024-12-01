import os
import pandas as pd
import struct

# Define the relative path to your log file
script_dir = os.getcwd()  # Use current working directory
log_file = os.path.join(script_dir, '..', 'azim30_elev30_static_log_2024-11-28.csv')

# Load the CSV file
data = pd.read_csv(log_file)

def parse_frame_header(raw_data):
    """
    Parse the first 40 bytes of raw data into a structured frame header.
    
    :param raw_data: List of integers from the RawData
    :return: Dictionary representing the frame header
    """
    # Ensure we have at least 40 bytes
    if len(raw_data) < 40:
        raise ValueError("Insufficient data for Frame Header")
    
    # Convert list of integers to bytes
    raw_bytes = bytes(raw_data[:40])
    
    # Unpack the bytes according to the Frame Header structure
    frame_header = struct.unpack('<QIIIIIIII', raw_bytes)
    
    # Map unpacked data to corresponding fields (all in hexadecimal format)
    header_dict = {
        "Magic Word": f"0x{frame_header[0]:016X}",  # 64-bit hex
        "Version": f"0x{frame_header[1]:08X}",     # 32-bit hex
        "Total Packet Length": f"0x{frame_header[2]:08X}",
        "Platform": f"0x{frame_header[3]:08X}",
        "Frame Number": f"0x{frame_header[4]:08X}",
        "Time [in CPU Cycles]": f"0x{frame_header[5]:08X}",
        "Num Detected Obj": f"0x{frame_header[6]:08X}",
        "Num TLVs": f"0x{frame_header[7]:08X}",
        "Subframe Number": f"0x{frame_header[8]:08X}"
    }
    return header_dict

# Process and print the Frame Header for the first row only
first_row = data.iloc[0]
raw_data_list = [int(x) for x in first_row['RawData'].split(',')]

# Parse the frame header
try:
    frame_header = parse_frame_header(raw_data_list)
    print("\nParsed Frame Header (Hexadecimal Format):")
    for key, value in frame_header.items():
        print(f"{key}: {value}")
except ValueError as e:
    print(f"Error: {e}")
