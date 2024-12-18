import os
import pandas as pd
import numpy as np
import struct

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

            # Initialize the frame entry
            frames_dict[frame_number] = {
                "Frame Header": frame_header,
                "Detected Points": [],
                "Type 1 Data": [],
                "Type 2 Data": [],
                "Type 3 Data": [],
                "Type 7 Data": []
            }

            # Parse TLVs
            for _ in range(num_tlvs):
                if len(raw_data_list) < 8:
                    print(f"Skipping incomplete TLV data in Frame {frame_number}")
                    break

                tlv_header = parse_tlv_header(raw_data_list)

                if tlv_header["TLV Type"] == 1:
                    # Detected Points
                    tlv_payload = parse_type_1_data(tlv_header, raw_data_list)
                    if tlv_payload and "Detected Points" in tlv_payload:
                        frames_dict[frame_number]["Detected Points"].extend(tlv_payload["Detected Points"])
                        frames_dict[frame_number]["Type 1 Data"].append(tlv_payload)

                elif tlv_header["TLV Type"] == 2:
                    # Type 2: Placeholder for additional data
                    tlv_payload = parse_type_2_data(tlv_header, raw_data_list)
                    frames_dict[frame_number]["Type 2 Data"].append(tlv_payload)

                elif tlv_header["TLV Type"] == 3:
                    # Type 3: Placeholder for another data type
                    tlv_payload = parse_type_3_data(tlv_header, raw_data_list)
                    frames_dict[frame_number]["Type 3 Data"].append(tlv_payload)

                elif tlv_header["TLV Type"] == 7:
                    # Type 7: Side Info for Detected Points
                    num_detected_obj = frame_header["Num Detected Obj"]  # Get number of detected objects
                    tlv_payload = parse_type_7_data(tlv_header, raw_data_list, num_detected_obj, frame_number)
                    frames_dict[frame_number]["Type 7 Data"].append(tlv_payload)


        except (ValueError, IndexError) as e:
            print(f"Error parsing row {row_idx + 1}: {e}")

    return frames_dict

def parse_type_1_data(tlv_header, raw_data_list):
    """Parses Type 1 TLV payload (Detected Points)."""
    # Extract data according to the Type 1 format
    payload_length = tlv_header["TLV Length"]
    point_size = 16
    payload_data = raw_data_list[:payload_length]
    raw_data_list[:payload_length] = []  # Remove parsed data

    detected_points = []
    for i in range(payload_length // point_size):
        point_bytes = bytes(payload_data[i * point_size:(i + 1) * point_size])
        x, y, z, doppler = struct.unpack('<ffff', point_bytes)
        detected_points.append({"X [m]": x, "Y [m]": y, "Z [m]": z, "Doppler [m/s]": doppler})

    return {"Detected Points": detected_points}

def parse_type_2_data(tlv_header, raw_data_list):
    """Parses Type 2 TLV payload. Placeholder function."""
    # Extract data according to the Type 2 format
    payload_length = tlv_header["TLV Length"]
    payload_data = raw_data_list[:payload_length]
    raw_data_list[:payload_length] = []  # Remove parsed data
    return {"Type 2 Payload": payload_data}

def parse_type_3_data(tlv_header, raw_data_list):
    """Parses Type 3 TLV payload. Placeholder function."""
    # Extract data according to the Type 3 format
    payload_length = tlv_header["TLV Length"]
    payload_data = raw_data_list[:payload_length]
    raw_data_list[:payload_length] = []  # Remove parsed data
    return {"Type 3 Payload": payload_data}

def parse_type_7_data(tlv_header, raw_data_list, num_detected_obj, frame_number):
    """Parses Type 7 TLV payload (Side Info for Detected Points)."""
    payload_length = tlv_header["TLV Length"]
    expected_length = 4 * num_detected_obj  # 4 bytes per point (2 for SNR, 2 for Noise)

    # Check if the payload length matches the expected length
    if payload_length != expected_length:
        print(f"Warning: Type 7 payload length mismatch. Expected {expected_length}, got {payload_length}. Skipping.")
        raw_data_list[:payload_length] = []  # Consume and skip invalid payload
        return {"Side Info": []}

    side_info = []
    for i in range(num_detected_obj-1):
        # Ensure we have enough bytes for each point
        if len(raw_data_list) < 4:
            print(f"Warning: Insufficient data for point {i} in Type 7 payload in Frame {frame_number}. Skipping remaining points.")
            break

        point_bytes = bytes(raw_data_list[:4])
        raw_data_list[:4] = []  # Remove parsed bytes

        snr, noise = struct.unpack('<HH', point_bytes)  # uint16_t for SNR and Noise
        side_info.append({
            "SNR [dB]": snr * 0.1,  # Convert to dB
            "Noise [dB]": noise * 0.1
        })

    return {"Side Info": side_info}




def extract_coordinates_with_doppler_and_side_info(frames_data):
    """
    Extracts coordinates with Doppler information from Type 1 TLV data and SNR/Noise from Type 7.

    Args:
        frames_data (dict): Parsed frame data from the log file.

    Returns:
        dict: A dictionary with frame numbers and their detected points, Doppler info, and side info.
    """
    coordinates_dict = {}

    for frame_num, frame_content in frames_data.items():
        points = frame_content.get("Detected Points", [])
        side_info_data = frame_content.get("Type 7 Data", [{}])
        side_info_list = side_info_data[0].get("Side Info", []) if side_info_data else []

        coordinates = []
        for i, point in enumerate(points):
            snr = side_info_list[i]["SNR [dB]"] if i < len(side_info_list) else None
            noise = side_info_list[i]["Noise [dB]"] if i < len(side_info_list) else None
            coordinates.append({
                "X [m]": point["X [m]"],
                "Y [m]": point["Y [m]"],
                "Z [m]": point["Z [m]"],
                "Doppler [m/s]": point["Doppler [m/s]"],
                "SNR [dB]": snr,
                "Noise [dB]": noise
            })

        coordinates_dict[frame_num] = coordinates

    return coordinates_dict

def visualize_tlv_data(frames_data):
    """
    Visualizes data for TLV Types 2, 3, and 7.

    Args:
        frames_data (dict): Parsed frame data from the log file.
    """
    for frame_num, frame_content in frames_data.items():
        print(f"\nFrame {frame_num}:")

        # Type 2 Data
        if frame_content["Type 2 Data"]:
            print("  Type 2 Data:")
            for entry in frame_content["Type 2 Data"]:
                print(f"    {entry}")

        # Type 3 Data
        if frame_content["Type 3 Data"]:
            print("  Type 3 Data:")
            for entry in frame_content["Type 3 Data"]:
                print(f"    {entry}")

        # Type 7 Data
        if frame_content["Type 7 Data"]:
            print("  Type 7 Data (Side Info):")
            for entry in frame_content["Type 7 Data"]:
                for info in entry["Side Info"]:
                    print(f"    SNR: {info['SNR [dB]']:.1f} dB, Noise: {info['Noise [dB]']:.1f} dB")



if __name__ == "__main__":
    # Set up the file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = os.path.join("..", "..", "..", "Logs", "LogsPart3", "StaticMonitoring", "Test_30fps_dist15mts_vehicleLog_5mps_d3x3wall_att1_log.csv")
    file_path = os.path.normpath(os.path.join(script_dir, relative_path))

    print(f"Processing file: {file_path}")
    frames_data = process_log_file(file_path)

    # Print summary
    print(f"\nParsed {len(frames_data)} frames successfully.\n")

    # Visualize Type 1 TLV data (coordinates and Doppler) with SNR and Noise for the first 5 frames
    print("Type 1 TLV Data (Coordinates, Doppler, SNR, and Noise) for the first 5 frames:")
    coordinates_dict = extract_coordinates_with_doppler_and_side_info(frames_data)
    for frame_num, points in list(coordinates_dict.items())[:5]:  # Show first 5 frames
        print(f"\nFrame {frame_num}:")
        for point in points:
            print(f"  X: {point['X [m]']:.3f}, Y: {point['Y [m]']:.3f}, Z: {point['Z [m]']:.3f}, Doppler: {point['Doppler [m/s]']:.3f}")
            if point["SNR [dB]"] is not None and point["Noise [dB]"] is not None:
                print(f"    SNR: {point['SNR [dB]']:.1f} dB, Noise: {point['Noise [dB]']:.1f} dB")
            else:
                print("    SNR: N/A, Noise: N/A")
