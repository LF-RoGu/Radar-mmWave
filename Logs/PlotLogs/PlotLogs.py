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
    payload_length = tlv_length  # Subtract header size

    if len(raw_data) < payload_length:
        raise ValueError(f"Insufficient data for TLV Payload: expected {payload_length} bytes, "
                         f"but got {len(raw_data)} bytes.")

    # Extract the payload as a list
    payload = [raw_data.pop(0) for _ in range(payload_length)]

    # Process the payload based on the TLV type
    if tlv_type == 1:  # Detected Points
        point_size = 16  # Each detected point is 16 bytes
        detected_points = []
        for i in range(payload_length // point_size):
            point_bytes = bytes(payload[i * point_size:(i + 1) * point_size])
            x, y, z, doppler = struct.unpack('<ffff', point_bytes)
            detected_points.append({"X [m]": x, "Y [m]": y, "Z [m]": z, "Doppler [m/s]": doppler})
        return {"Detected Points": detected_points}

    elif tlv_type in (2, 3):  # Range Profile or Noise Profile
        range_points = []
        for i in range(payload_length // 2):  # Each point is 2 bytes
            point_raw = (payload[i * 2 + 1] << 8) | payload[i * 2]
            point_q9 = point_raw / 512.0  # Convert Q9 format to float
            range_points.append(point_q9)
        return {"Range Profile" if tlv_type == 2 else "Noise Profile": range_points}

    elif tlv_type in (4, 8):  # Azimuth Static Heatmap or Azimuth/Elevation Heatmap
        heatmap = []
        for i in range(payload_length // 4):  # Each complex number is 4 bytes
            imag = (payload[i * 4 + 1] << 8) | payload[i * 4]
            real = (payload[i * 4 + 3] << 8) | payload[i * 4 + 2]
            heatmap.append({"Real": real, "Imaginary": imag})
        return {"Azimuth Static Heatmap" if tlv_type == 4 else "Azimuth/Elevation Static Heatmap": heatmap}

    elif tlv_type == 5:  # Range-Doppler Heatmap
        heatmap = []
        row_size = int(payload_length ** 0.5)  # Assuming square 2D array
        for i in range(row_size):
            row = payload[i * row_size:(i + 1) * row_size]
            heatmap.append(row)
        return {"Range-Doppler Heatmap": heatmap}

    elif tlv_type == 6:  # Statistics
        stats = struct.unpack('<' + 'I' * (payload_length // 4), bytes(payload))
        return {
            "Statistics": {
                "InterFrameProcessingTime": stats[0],
                "TransmitOutputTime": stats[1],
                "InterFrameProcessingMargin": stats[2],
                "InterChirpProcessingMargin": stats[3],
                "ActiveFrameCPULoad": stats[4],
                "InterFrameCPULoad": stats[5]
            }
        }

    elif tlv_type == 7:  # Side Info for Detected Points
        side_info = []
        point_size = 4  # Each point has 4 bytes of side info
        for i in range(payload_length // point_size):
            snr, noise = struct.unpack('<HH', bytes(payload[i * point_size:(i + 1) * point_size]))
            side_info.append({"SNR": snr, "Noise": noise})
        return {"Side Info for Detected Points": side_info}

    elif tlv_type == 9:  # Temperature Statistics
        # Type 9 payload structure:
        # 4 bytes: TempReportValid (uint32_t)
        # 4 bytes: Time (uint32_t)
        # 2 bytes each: Remaining temperature values (uint16_t)
        if payload_length != 28:
            raise ValueError(f"Invalid payload length for Type 9: expected 28 bytes, got {payload_length} bytes")

        # Parse the payload manually
        temp_report_valid = (payload[3] << 24) | (payload[2] << 16) | (payload[1] << 8) | payload[0]
        time_ms = (payload[7] << 24) | (payload[6] << 16) | (payload[5] << 8) | payload[4]

        temperatures = []
        for i in range(8, payload_length, 2):  # Start at index 8, step by 2 for uint16_t
            temp = (payload[i + 1] << 8) | payload[i]
            temperatures.append(temp)

        # Map temperatures to sensor names
        sensor_names = [
            "TmpRx0Sens", "TmpRx1Sens", "TmpRx2Sens", "TmpRx3Sens",
            "TmpTx0Sens", "TmpTx1Sens", "TmpTx2Sens", "TmpPmSens",
            "TmpDig0Sens", "TmpDig1Sens"
        ]
        temperature_data = dict(zip(sensor_names, temperatures))

        return {
            "Temperature Statistics": {
                "TempReportValid": temp_report_valid,
                "Time (ms)": time_ms,
                **temperature_data
            }
        }


    else:
        # Unknown TLV type
        return {"Raw Payload": payload}

def print_tlvs(num_tlvs, raw_data_list):
    """
    Parse and print all TLVs for the current frame.
    :param num_tlvs: Number of TLVs to process (from Frame Header).
    :param raw_data_list: List of raw data to parse.
    """
    for tlv_idx in range(num_tlvs):
        print(f"\n--- Parsing TLV {tlv_idx + 1} ---")

        # Parse the TLV header
        tlv_header = parse_tlv_header(raw_data_list)
        print("\nParsed TLV Header:")
        for key, value in tlv_header.items():
            print(f"{key}: {value}")

        # Parse the TLV payload
        tlv_payload = parse_tlv_payload(tlv_header, raw_data_list)
        print("\nParsed TLV Payload:")
        print(tlv_payload)

        # Debug: Print remaining raw_data_list before parsing next TLV
        print(f"\nRemaining Data After First TLV: {raw_data_list[:20]}...")  # Print first 20 bytes

    return



if __name__ == "__main__":
    # Define the relative path to your log file
    script_dir = os.getcwd()  # Use current working directory
    log_file = os.path.join(script_dir, '..', 'azim30_elev30_static_log_2024-11-28.csv')

    # Load the CSV file
    data = pd.read_csv(log_file)

    # Ask the user for the number of rows to process
    try:
        num_rows_to_process = int(input("Enter the number of rows to process: "))
    except ValueError:
        print("Invalid input. Please enter an integer value.")
        exit(1)

    # Process the specified number of rows
    for row_idx in range(min(num_rows_to_process, len(data))):
        print(f"\n--- Processing Row {row_idx + 1} ---")
        try:
            # Get raw data from the current row
            raw_data_list = [int(x) for x in data.iloc[row_idx]['RawData'].split(',')]

            # Parse the frame header
            frame_header = parse_frame_header(raw_data_list)
            print("\nParsed Frame Header:")
            for key, value in frame_header.items():
                print(f"{key}: {value}")

            # Parse and print TLVs using the extracted number of TLVs from the frame header
            num_tlvs = frame_header["Num TLVs"]
            print_tlvs(num_tlvs, raw_data_list)

        except Exception as e:
            print(f"Error processing row {row_idx + 1}: {e}")
