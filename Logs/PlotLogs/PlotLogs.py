import os
import pandas as pd
import struct
import matplotlib.pyplot as plt
from datetime import datetime

# Global variable to specify which TLVs to process
interested_tlv_types = [1]  # Example: Interested in Detected Points (1) and Temperature Statistics (9)

def initialize_csv(filename="coordinates.csv"):
    """
    Initializes the CSV file by deleting any existing data and adding headers.
    :param filename: Name of the CSV file to initialize.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Current script directory
    file_path = os.path.join(script_dir, filename)

    # Create or overwrite the file with headers
    with open(file_path, 'w') as f:
        f.write("Frame,Timestamp,X [m],Y [m],Z [m],Doppler [m/s]\n")

    print(f"Initialized CSV at {file_path}")
    return file_path

def append_frame_to_csv(frame_number, timestamp, coordinates, filename="coordinates.csv"):
    """
    Appends frame data to the CSV file.
    :param frame_number: The frame number.
    :param timestamp: The UNIX timestamp of the frame.
    :param coordinates: List of dictionaries with keys "X [m]", "Y [m]", "Z [m]", "Doppler [m/s]".
    :param filename: Name of the CSV file to append to.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Current script directory
    file_path = os.path.join(script_dir, filename)

    # Append the frame data
    with open(file_path, 'a') as f:
        for point in coordinates:
            f.write(f"{frame_number},{timestamp},{point['X [m]']},{point['Y [m]']},{point['Z [m]']},{point['Doppler [m/s]']}\n")

    #print(f"Appended Frame {frame_number} data to {file_path}")


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

        #Insert logic here

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

def convert_timestamp_to_unix(timestamp_str):
    """
    Convert a timestamp string into UNIX format, handling nanoseconds by truncating them.
    """
    try:
        # Split the timestamp to remove excess precision (nanoseconds)
        truncated_timestamp = timestamp_str.split('.')[0] + '.' + timestamp_str.split('.')[1][:6]
        # Parse the truncated timestamp
        dt = datetime.strptime(truncated_timestamp, '%Y-%m-%d %H:%M:%S.%f')
        # Convert to UNIX timestamp (seconds since epoch)
        return dt.timestamp()
    except ValueError as e:
        print(f"Error parsing timestamp '{timestamp_str}': {e}")
        return None

def plot_all_data(data, doppler_threshold=0.1, axis_limit=3):
    """
    Plot all stationary and moving objects in a single frame without erasing any points.
    """
    csv_file = initialize_csv()  # Call the CSV initializer if logging is enabled

    stationary_coords = []  # Stationary points: (X, Y, Z)
    moving_coords = []      # Moving points: (X, Y, Z)

    for row_idx in range(len(data)):
        try:
            # Check if the row is valid (non-null)
            if pd.isnull(data.iloc[row_idx]['Timestamp']) or pd.isnull(data.iloc[row_idx]['RawData']):
                print(f"Skipping row {row_idx + 1}: Null data encountered.")
                continue

            # Convert timestamp to UNIX format
            timestamp = convert_timestamp_to_unix(data.iloc[row_idx]['Timestamp'])
            if timestamp is None:
                print(f"Skipping row {row_idx + 1} due to invalid timestamp.")
                continue

            # Get raw data from the current row
            raw_data_list = [int(x) for x in data.iloc[row_idx]['RawData'].split(',')]

            # Parse the frame header
            frame_header = parse_frame_header(raw_data_list)
            num_tlvs = frame_header["Num TLVs"]

            # Use the row index as the frame number (1-indexed)
            frame_number = row_idx + 1

            # Parse TLVs
            for _ in range(num_tlvs):
                tlv_header = parse_tlv_header(raw_data_list)
                if tlv_header["TLV Type"] == 1:  # Interested in Detected Points
                    tlv_payload = parse_tlv_payload(tlv_header, raw_data_list)
                    if tlv_payload and "Detected Points" in tlv_payload:
                        for point in tlv_payload["Detected Points"]:
                            # Categorize based on Doppler threshold
                            if abs(point["Doppler [m/s]"]) <= doppler_threshold:
                                stationary_coords.append((point["X [m]"], point["Y [m]"], point["Z [m]"]))
                            else:
                                moving_coords.append((point["X [m]"], point["Y [m]"], point["Z [m]"]))

                        append_frame_to_csv(frame_number, timestamp, tlv_payload["Detected Points"], csv_file)

        except Exception as e:
            print(f"Error processing row {row_idx + 1}: {e}")

    # Plot stationary and moving objects
    fig = plt.figure(figsize=(12, 6))
    ax_stationary = fig.add_subplot(121, projection='3d', title="Stationary Objects")
    ax_moving = fig.add_subplot(122, projection='3d', title="Moving Objects")

    # Plot stationary objects
    if stationary_coords:
        x_stationary, y_stationary, z_stationary = zip(*stationary_coords)
        ax_stationary.scatter(x_stationary, y_stationary, z_stationary, c='green', marker='o')
    ax_stationary.set_xlabel('X Coordinate (m)')
    ax_stationary.set_ylabel('Y Coordinate (m)')
    ax_stationary.set_zlabel('Z Coordinate (m)')
    ax_stationary.set_xlim([-axis_limit, axis_limit])
    ax_stationary.set_ylim([-axis_limit, axis_limit])
    ax_stationary.set_zlim([-axis_limit, axis_limit])


    # Plot moving objects
    if moving_coords:
        x_moving, y_moving, z_moving = zip(*moving_coords)
        ax_moving.scatter(x_moving, y_moving, z_moving, c='red', marker='o')
    ax_moving.set_xlabel('X Coordinate (m)')
    ax_moving.set_ylabel('Y Coordinate (m)')
    ax_moving.set_zlabel('Z Coordinate (m)')
    ax_moving.set_xlim([-axis_limit, axis_limit])
    ax_moving.set_ylim([-axis_limit, axis_limit])
    ax_moving.set_zlim([-axis_limit, axis_limit])

    plt.tight_layout()
    plt.show()

def live_visualization(data, doppler_threshold=0.1, axis_limit=3, delay=0.5):
    """
    Live visualization of stationary and moving objects with updates based on timestamp.
    """
    fig = plt.figure(figsize=(12, 6))
    ax_stationary = fig.add_subplot(121, projection='3d')
    ax_moving = fig.add_subplot(122, projection='3d')

    for row_idx in range(len(data)):
        try:
            # Check if the row is valid (non-null)
            if pd.isnull(data.iloc[row_idx]['Timestamp']) or pd.isnull(data.iloc[row_idx]['RawData']):
                print(f"Stopping processing at row {row_idx + 1}: Null data encountered.")
                break

            # Convert timestamp to UNIX format
            timestamp = convert_timestamp_to_unix(data.iloc[row_idx]['Timestamp'])
            if timestamp is None:
                print(f"Skipping row {row_idx + 1} due to invalid timestamp.")
                continue

            # Get raw data from the current row
            raw_data_list = [int(x) for x in data.iloc[row_idx]['RawData'].split(',')]

            # Parse the frame header
            frame_header = parse_frame_header(raw_data_list)
            num_tlvs = frame_header["Num TLVs"]

            # Storage for stationary and moving points
            stationary_coords = []  # Stationary points: (X, Y, Z)
            moving_coords = []      # Moving points: (X, Y, Z)

            # Parse TLVs
            for _ in range(num_tlvs):
                tlv_header = parse_tlv_header(raw_data_list)
                if tlv_header["TLV Type"] == 1:  # Interested in Detected Points
                    tlv_payload = parse_tlv_payload(tlv_header, raw_data_list)
                    if tlv_payload and "Detected Points" in tlv_payload:
                        for point in tlv_payload["Detected Points"]:
                            # Categorize based on Doppler threshold
                            if abs(point["Doppler [m/s]"]) <= doppler_threshold:
                                stationary_coords.append((point["X [m]"], point["Y [m]"], point["Z [m]"]))
                            else:
                                moving_coords.append((point["X [m]"], point["Y [m]"], point["Z [m]"]))

            # Clear plots for new frame
            ax_stationary.cla()
            ax_moving.cla()

            # Update stationary plot
            if stationary_coords:
                x_stationary, y_stationary, z_stationary = zip(*stationary_coords)
                ax_stationary.scatter(x_stationary, y_stationary, z_stationary, c='green', marker='o')
            ax_stationary.set_xlim([-axis_limit, axis_limit])
            ax_stationary.set_ylim([-axis_limit, axis_limit])
            ax_stationary.set_zlim([-axis_limit, axis_limit])
            ax_stationary.set_xlabel('X Coordinate (m)')
            ax_stationary.set_ylabel('Y Coordinate (m)')
            ax_stationary.set_zlabel('Z Coordinate (m)')
            ax_stationary.set_title("Stationary Objects")

            # Update moving plot
            if moving_coords:
                x_moving, y_moving, z_moving = zip(*moving_coords)
                ax_moving.scatter(x_moving, y_moving, z_moving, c='red', marker='o')
            ax_moving.set_xlim([-axis_limit, axis_limit])
            ax_moving.set_ylim([-axis_limit, axis_limit])
            ax_moving.set_zlim([-axis_limit, axis_limit])
            ax_moving.set_xlabel('X Coordinate (m)')
            ax_moving.set_ylabel('Y Coordinate (m)')
            ax_moving.set_zlabel('Z Coordinate (m)')
            ax_moving.set_title("Moving Objects")

            # Pause to simulate live update with delay
            plt.pause(delay)

        except Exception as e:
            print(f"Error processing row {row_idx + 1}: {e}")

    plt.show()

if __name__ == "__main__":
    # Define the relative path to your log file
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Current script directory
    parent_dir = os.path.dirname(script_dir)  # One directory above the current directory
    log_file = os.path.join(parent_dir, 'LogsPart4', 'Test_30fps_dist15mts_vehicleLog_log.csv')

    # Load the CSV file
    data = pd.read_csv(log_file)

    # Choose mode of visualization
    mode = input("Enter 'live' (1) for live visualization or 'all' (2) to plot all data: ").strip().lower()

    if mode in ['live', '1']:
        # Run live visualization
        live_visualization(data, doppler_threshold=0.1, axis_limit=12, delay=0.1)
    elif mode in ['all', '2']:
        # Plot all data
        plot_all_data(data, doppler_threshold=0.1, axis_limit=12)
    else:
        print("Invalid mode. Please enter 'live' or 'all'.")
