import os
import pandas as pd
import struct
from datetime import datetime

#Only decodeData should be visible to the outside
__all__ = ['decodeData']

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

    # Process TLVs based on type
    if tlv_type == 1:  # Detected Points
        point_size = 16
        detected_points = []
        for i in range(payload_length // point_size):
            point_bytes = bytes(payload[i * point_size:(i + 1) * point_size])
            x, y, z, doppler = struct.unpack('<ffff', point_bytes)
            detected_points.append({"x": x, "y": y, "z": z, "doppler": doppler})
        return {"detectedPoints": detected_points}

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
            side_info.append({"snr": snr, "noise": noise})
        return {"SNRandNoise": side_info}

    elif tlv_type == 9:  # Temperature Statistics
        if payload_length != 28:
            raise ValueError(f"Invalid payload length for Type 9: expected 28 bytes, got {payload_length} bytes")

        temp_report_valid = (payload[3] << 24) | (payload[2] << 16) | (payload[1] << 8) | payload[0]
        time_ms = (payload[7] << 24) | (payload[6] << 16) | (payload[5] << 8) | payload[4]

        temperatures = []
        for i in range(8, payload_length, 2):  # Start at index 8, step by 2 for uint16_t
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

    # Default case for unknown TLV types
    return {"Unknown TLV Type": tlv_type, "Payload": payload}


def dataToFrames(raw_frame):
    #Preparing a return list containing all frames
    decodedFrames = []
    raw_data_list = list(raw_frame)
    
    try:
        #Parsing the frame header
        frame_header = parse_frame_header(raw_data_list)
        num_tlvs = frame_header["Num TLVs"]

        decodedFrame = {}
        decodedSNRandNoise = None

        # Parse TLVs
        for _ in range(num_tlvs):
            tlv_header = parse_tlv_header(raw_data_list)
            if tlv_header["TLV Type"] == 1:  # Interested in Detected Points
                tlv_payload = parse_tlv_payload(tlv_header, raw_data_list)
                if tlv_payload and "detectedPoints" in tlv_payload:
                    decodedFrame = {"detectedPoints":tlv_payload["detectedPoints"]}
            elif tlv_header["TLV Type"] == 7: # Interested in Side Info of Detected Points
                tlv_payload = parse_tlv_payload(tlv_header, raw_data_list)
                if tlv_payload and "SNRandNoise" in tlv_payload:
                    decodedSNRandNoise = tlv_payload["SNRandNoise"]
        
        #Adding SNR and noise info to the decoded frames
        if "detectedPoints" in decodedFrame and decodedSNRandNoise:
            for i, point in enumerate(decodedFrame["detectedPoints"]):
                point["snr"] = decodedSNRandNoise[i]["snr"]
                point["noise"] = decodedSNRandNoise[i]["noise"]

        #Adding the decoded frame to the list of decoded frames
        decodedFrames.append(decodedFrame)

    except Exception as e:
        print(f"Error decoding frame: {e}")

    return decodedFrames

def decodeData(file_path):
    csvData = pd.read_csv(file_path)
    decodedFrames = dataToFrames(csvData)
    return decodedFrames