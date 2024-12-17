import os
import pandas as pd
import struct
import csv

# Parse Frame Header
def parse_frame_header(raw_data):
    if len(raw_data) < 40:
        raise ValueError("Insufficient data for Frame Header")
    raw_bytes = bytes([raw_data.pop(0) for _ in range(40)])
    frame_header = struct.unpack('<QIIIIIIII', raw_bytes)
    return {
        "Magic Word": f"0x{frame_header[0]:016X}",
        "Version": f"0x{frame_header[1]:08X}",
        "Total Packet Length": frame_header[2],
        "Platform": f"0x{frame_header[3]:08X}",
        "Frame Number": frame_header[4],
        "Time [in CPU Cycles]": frame_header[5],
        "Num Detected Obj": frame_header[6],
        "Num TLVs": frame_header[7],
        "Subframe Number": frame_header[8]
    }

# Parse TLV Header
def parse_tlv_header(raw_data):
    if len(raw_data) < 8:
        raise ValueError("Insufficient data for TLV Header")
    raw_bytes = bytes([raw_data.pop(0) for _ in range(8)])
    tlv_type, tlv_length = struct.unpack('<II', raw_bytes)
    return {"TLV Type": tlv_type, "TLV Length": tlv_length}

# Parse TLV Payload
def parse_tlv_payload(tlv_header, raw_data):
    tlv_type = tlv_header["TLV Type"]
    payload_length = tlv_header["TLV Length"]
    payload = [raw_data.pop(0) for _ in range(payload_length)]

    # Detected Points Example
    if tlv_type == 1:  # Detected Points
        point_size = 16
        detected_points = []
        for i in range(payload_length // point_size):
            point_bytes = bytes(payload[i * point_size:(i + 1) * point_size])
            x, y, z, doppler = struct.unpack('<ffff', point_bytes)
            detected_points.append({"X [m]": x, "Y [m]": y, "Z [m]": z, "Doppler [m/s]": doppler})
        return {"Detected Points": detected_points}
    return None

# Process the CSV file and parse data
def process_log_file(file_path):
    """
    Parses the log file and returns all frames and detected points as a dictionary.
    
    Returns:
        dict: A dictionary containing frame headers and their respective detected points.
    """
    frames_dict = {}  # Dictionary to hold all parsed frame data

    # Load the CSV data, skip the header row
    data = pd.read_csv(file_path, names=["Timestamp", "RawData"], skiprows=1)

    for row_idx in range(len(data)):
        try:
            # Skip invalid rows
            if pd.isnull(data.iloc[row_idx]['RawData']):
                print(f"Skipping row {row_idx + 1}: Invalid or null data.")
                continue

            # Convert raw data to a list of integers
            raw_data_list = [int(x) for x in data.iloc[row_idx]['RawData'].split(',')]

            # Parse the Frame Header
            frame_header = parse_frame_header(raw_data_list)
            num_tlvs = frame_header["Num TLVs"]
            frame_number = frame_header["Frame Number"]
            #print(f"Parsing Frame {frame_number}: {frame_header}")

            # Initialize the frame entry
            frames_dict[frame_number] = {
                "Frame Header": frame_header,
                "Detected Points": []
            }

            # Parse TLVs
            for _ in range(num_tlvs):
                if len(raw_data_list) < 8:
                    print(f"Skipping incomplete TLV data in Frame {frame_number}")
                    break
                
                tlv_header = parse_tlv_header(raw_data_list)

                # Only process Detected Points (TLV Type 1)
                if tlv_header["TLV Type"] == 1:
                    tlv_payload = parse_tlv_payload(tlv_header, raw_data_list)
                    if tlv_payload and "Detected Points" in tlv_payload:
                        frames_dict[frame_number]["Detected Points"].extend(tlv_payload["Detected Points"])

        except (ValueError, IndexError) as e:
            print(f"Error parsing row {row_idx + 1}: {e}")

    return frames_dict

# Create a new dictionary with frame numbers and coordinates + Doppler speed
def extract_coordinates_with_doppler(frames_data):
    coordinates_dict = {}
    
    for frame_num, frame_content in frames_data.items():
        # Extract detected points for the current frame
        points = frame_content["Detected Points"]
        
        # Create a list of dictionaries with required fields
        coordinates = [
            {
                "X [m]": point["X [m]"],
                "Y [m]": point["Y [m]"],
                "Z [m]": point["Z [m]"],
                "Doppler [m/s]": point["Doppler [m/s]"]
            }
            for point in points
        ]
        
        # Add the coordinates list to the new dictionary under the frame number
        coordinates_dict[frame_num] = coordinates
    
    return coordinates_dict



# Main script
if __name__ == "__main__":
    # Set up the file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = os.path.join("..", "..", "..", "Logs", "LogsPart3", "DynamicMonitoring", "30fps_straight_3x3_3_log_2024-12-16.csv")
    file_path = os.path.normpath(os.path.join(script_dir, relative_path))

    print(f"Processing file: {file_path}")
    frames_data = process_log_file(file_path)

    # Print summary
    print(f"\nParsed {len(frames_data)} frames successfully.")

    # Extract new dictionary with frame numbers and coordinates + Doppler
    coordinates_dict = extract_coordinates_with_doppler(frames_data)

    # Print the first 3 frames as a preview
    print("\nCoordinates with Doppler speed for the first 3 frames:")
    for frame_num, points in list(coordinates_dict.items())[:50]:  # Show first 3 frames
        print(f"\nFrame {frame_num}:")
        for point in points[:5]:  # Show first 5 points
            print(f"  X: {point['X [m]']:.3f}, Y: {point['Y [m]']:.3f}, Z: {point['Z [m]']:.3f}, Doppler: {point['Doppler [m/s]']:.3f}")