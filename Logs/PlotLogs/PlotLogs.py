import os
import pandas as pd
import struct


def parse_frame_header(raw_data):
    """
    Parse the first 40 bytes of raw data into a structured frame header.
    Values are popped from the raw_data list as they are parsed.
    """
    if len(raw_data) < 40:
        raise ValueError("Insufficient data for Frame Header")

    # Extract values and unpack
    raw_bytes = bytes([raw_data.pop(0) for _ in range(40)])
    frame_header = struct.unpack('<QIIIIIIII', raw_bytes)

    header_dict = {
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
    return header_dict


def parse_tlv_header(raw_data):
    """
    Parse the next 8 bytes of raw data as the TLV header.
    Values are popped from the raw_data list as they are parsed.
    """
    if len(raw_data) < 8:
        raise ValueError("Insufficient data for TLV Header")

    # Extract values and unpack
    raw_bytes = bytes([raw_data.pop(0) for _ in range(8)])
    tlv_type, tlv_length = struct.unpack('<II', raw_bytes)

    header_dict = {
        "TLV Type": tlv_type,
        "TLV Length": tlv_length
    }
    return header_dict


def parse_tlv_payload(tlv_header, raw_data):
    """
    Parse the TLV payload based on the TLV header type.
    Values are popped from the raw_data list as they are parsed.
    """
    tlv_type = tlv_header["TLV Type"]
    tlv_length = tlv_header["TLV Length"]
    payload_length = tlv_length - 8  # Subtract header size

    if len(raw_data) < payload_length:
        raise ValueError(f"Insufficient data for TLV Payload: expected {payload_length} bytes, "
                         f"but got {len(raw_data)} bytes.")

    # Extract the payload as a list
    payload = [raw_data.pop(0) for _ in range(payload_length)]

    # Process the payload based on the TLV type
    if tlv_type == 1:
        # Detected Points
        point_size = 16  # Each detected point is 16 bytes
        detected_points = []
        for i in range(payload_length // point_size):
            point_bytes = bytes(payload[i * point_size:(i + 1) * point_size])
            x, y, z, doppler = struct.unpack('<ffff', point_bytes)
            detected_points.append({"X [m]": x, "Y [m]": y, "Z [m]": z, "Doppler [m/s]": doppler})
        return {"Detected Points": detected_points}

    elif tlv_type == 2:
        # Range Profile
        range_points = []
        for i in range(payload_length // 2):  # Each point is 2 bytes
            point_raw = (payload[i * 2 + 1] << 8) | payload[i * 2]  # Combine 2 bytes into 16-bit value
            point_q9 = point_raw / 512.0  # Convert Q9 format to float
            range_points.append(point_q9)
        return {"Range Profile": range_points}

    elif tlv_type == 3:
        # Noise Profile
        return {"Noise Profile": payload}

    elif tlv_type == 4:
        # Azimuth Static Heatmap
        return {"Azimuth Static Heatmap": payload}

    elif tlv_type == 5:
        # Doppler Static Heatmap
        return {"Doppler Static Heatmap": payload}

    elif tlv_type == 6:
        # Statistics
        return {"Statistics": payload}

    elif tlv_type == 7:
        # Side Info for Detected Points
        return {"Side Info": payload}

    elif tlv_type == 8:
        # Static Heatmap
        return {"Static Heatmap": payload}

    elif tlv_type == 9:
        # Temperature Statistics
        return {"Temperature Statistics": payload}

    else:
        # Unknown Type
        return {"Unknown Type": payload}


# Main code to read the file and parse data
if __name__ == "__main__":
    # Define the relative path to your log file
    script_dir = os.getcwd()  # Use current working directory
    log_file = os.path.join(script_dir, '..', 'azim30_elev30_static_log_2024-11-28.csv')

    # Load the CSV file
    data = pd.read_csv(log_file)

    try:
        # Process the first row of the CSV file
        first_row = data.iloc[0]
        raw_data_list = [int(x) for x in first_row['RawData'].split(',')]

        # Parse the frame header
        frame_header = parse_frame_header(raw_data_list)
        print("\nParsed Frame Header:")
        for key, value in frame_header.items():
            print(f"{key}: {value}")

        # Parse the first TLV header
        tlv_header = parse_tlv_header(raw_data_list)
        print("\nParsed TLV Header:")
        for key, value in tlv_header.items():
            print(f"{key}: {value}")

        # Parse the TLV payload
        tlv_payload = parse_tlv_payload(tlv_header, raw_data_list)
        print("\nParsed TLV Payload:")
        print(tlv_payload)

        # Remaining raw_data_list after parsing
        print(f"\nRemaining Raw Data: {raw_data_list}")

    except Exception as e:
        print(f"Error: {e}")
