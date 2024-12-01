import os
import pandas as pd
import struct

# Global variable to specify which TLVs to process
interested_tlv_types = [1]  # Example: Interested in Detected Points (1) and Temperature Statistics (9)

def parse_frame_header(raw_data):
    if len(raw_data) < 40:
        raise ValueError("Insufficient data for Frame Header")

    # Extract values and unpack
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

def parse_tlv_header(raw_data):
    if len(raw_data) < 8:
        raise ValueError("Insufficient data for TLV Header")

    # Extract values and unpack
    raw_bytes = bytes([raw_data.pop(0) for _ in range(8)])
    tlv_type, tlv_length = struct.unpack('<II', raw_bytes)

    return {"TLV Type": tlv_type, "TLV Length": tlv_length}

def parse_tlv_payload(tlv_header, raw_data):
    tlv_type = tlv_header["TLV Type"]
    tlv_length = tlv_header["TLV Length"]
    payload_length = tlv_length

    if len(raw_data) < payload_length:
        raise ValueError(f"Insufficient data for TLV Payload: expected {payload_length} bytes, "
                         f"but got {len(raw_data)} bytes.")

    # Extract the payload as a list
    payload = [raw_data.pop(0) for _ in range(payload_length)]

    # Only process TLVs we're interested in
    if tlv_type == 1:  # Detected Points
        point_size = 16
        detected_points = []
        for i in range(payload_length // point_size):
            point_bytes = bytes(payload[i * point_size:(i + 1) * point_size])
            x, y, z, doppler = struct.unpack('<ffff', point_bytes)
            detected_points.append({"X [m]": x, "Y [m]": y, "Z [m]": z, "Doppler [m/s]": doppler})
        return {"Detected Points": detected_points}

    elif tlv_type == 9:  # Temperature Statistics
        if payload_length != 28:
            raise ValueError(f"Invalid payload length for Type 9: expected 28 bytes, got {payload_length} bytes")

        temp_report_valid = (payload[3] << 24) | (payload[2] << 16) | (payload[1] << 8) | payload[0]
        time_ms = (payload[7] << 24) | (payload[6] << 16) | (payload[5] << 8) | payload[4]

        temperatures = []
        for i in range(8, payload_length, 2):
            temp = (payload[i + 1] << 8) | payload[i]
            temperatures.append(temp)

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

    # If not interested, return None
    return None

def print_tlvs(num_tlvs, raw_data_list):
    """
    Parse and print only the TLVs of interest for the current frame.
    """
    for tlv_idx in range(num_tlvs):
        # Parse the TLV header
        tlv_header = parse_tlv_header(raw_data_list)

        # Check if this TLV is in the list of types we're interested in
        if tlv_header["TLV Type"] in interested_tlv_types:
            print(f"\n--- Parsing TLV {tlv_idx + 1} ---")
            print("\nParsed TLV Header:")
            for key, value in tlv_header.items():
                print(f"{key}: {value}")

            # Parse the TLV payload
            tlv_payload = parse_tlv_payload(tlv_header, raw_data_list)
            if tlv_payload:
                print("\nParsed TLV Payload:")
                print(tlv_payload)
        else:
            # Skip payload for uninterested TLVs
            tlv_length = tlv_header["TLV Length"]

            # Ensure the list has enough data to discard
            if tlv_length > len(raw_data_list):
                print(f"WARNING: Insufficient data to discard uninterested TLV {tlv_idx + 1} payload. "
                      f"Expected {tlv_length} bytes, but only {len(raw_data_list)} bytes remain.")
                break  # Exit processing if there is insufficient data

            # Discard payload for uninterested TLVs
            if tlv_length > 0:
                _ = [raw_data_list.pop(0) for _ in range(tlv_length)]

    return



if __name__ == "__main__":
    # Define the relative path to your log file
    script_dir = os.getcwd()
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
