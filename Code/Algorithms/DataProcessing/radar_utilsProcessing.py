import struct
import math
import pandas as pd

class FrameHeader:
    def __init__(self, raw_data_list):
        if len(raw_data_list) < 40:
            raise ValueError("Insufficient data for Frame Header")
        raw_bytes = bytes([raw_data_list.pop(0) for _ in range(40)])
        frame_header = struct.unpack('<QIIIIIIII', raw_bytes)
        self.magic_word = f"0x{frame_header[0]:016X}"
        self.version = f"0x{frame_header[1]:08X}"
        self.total_packet_length = frame_header[2]
        self.platform = f"0x{frame_header[3]:08X}"
        self.frame_number = frame_header[4]
        self.time_cpu_cycles = frame_header[5]
        self.num_detected_obj = frame_header[6]
        self.num_tlvs = frame_header[7]
        self.subframe_number = frame_header[8]

class TLVHeader:
    def __init__(self, raw_data_list):
        if len(raw_data_list) < 8:
            raise ValueError("Insufficient data for TLV Header")
        raw_bytes = bytes([raw_data_list.pop(0) for _ in range(8)])
        self.tlv_type, self.tlv_length = struct.unpack('<II', raw_bytes)

class DetectedPoint:
    def __init__(self, x, y, z, doppler):
        self.x = x
        self.y = y
        self.z = z
        self.doppler = doppler
        self.range = math.sqrt(x**2 + y**2 + z**2)
        self.azimuth = math.degrees(math.atan2(x, y))
        self.elevation = math.degrees(math.atan2(z, math.sqrt(x**2 + y**2)))

class Type1Data:
    def __init__(self, tlv_header, raw_data_list):
        payload_length = tlv_header.tlv_length
        point_size = 16
        num_points = payload_length // point_size
        self.coordinates = []

        for _ in range(num_points):
            if len(raw_data_list) < point_size:
                break
            point_bytes = bytes([raw_data_list.pop(0) for _ in range(point_size)])
            x, y, z, doppler = struct.unpack('<ffff', point_bytes)
            self.coordinates.append(DetectedPoint(x, y, z, doppler))

class SideInfo:
    def __init__(self, snr, noise):
        self.snr = snr * 0.1
        self.noise = noise * 0.1

class Type7Data:
    def __init__(self, tlv_header, raw_data_list, num_detected_obj):
        payload_length = tlv_header.tlv_length
        expected_length = 4 * num_detected_obj
        self.side_info = []

        if payload_length != expected_length:
            raw_data_list[:payload_length] = []
            return

        for _ in range(num_detected_obj):
            if len(raw_data_list) < 4:
                break
            point_bytes = bytes([raw_data_list.pop(0) for _ in range(4)])
            snr, noise = struct.unpack('<HH', point_bytes)
            self.side_info.append(SideInfo(snr, noise))

class Frame:
    def __init__(self, frame_header):
        self.header = frame_header
        self.tlvs = []

class RadarLogParser:
    def __init__(self, file_path, snr_threshold=15.0):
        self.file_path = file_path
        self.snr_threshold = snr_threshold
        self.frames = []

    def process_log_file(self):
        data = pd.read_csv(self.file_path, names=["Timestamp", "RawData"], skiprows=1)

        for row_idx in range(len(data)):
            try:
                if pd.isnull(data.iloc[row_idx]['RawData']):
                    continue
                raw_data_list = [int(x) for x in data.iloc[row_idx]['RawData'].split(',')]
                frame_header = FrameHeader(raw_data_list)
                frame = Frame(frame_header)
                valid_frame = True

                for _ in range(frame_header.num_tlvs):
                    tlv_header = TLVHeader(raw_data_list)
                    if tlv_header.tlv_type == 1:
                        frame.tlvs.append(Type1Data(tlv_header, raw_data_list))
                    elif tlv_header.tlv_type == 7:
                        type7_data = Type7Data(tlv_header, raw_data_list, frame_header.num_detected_obj)
                        frame.tlvs.append(type7_data)

                        # Check if any SNR value is below the threshold
                        if any(info.snr < self.snr_threshold for info in type7_data.side_info):
                            valid_frame = False
                            break
                    else:
                        raw_data_list[:tlv_header.tlv_length] = []

                # Append frame only if it passes the SNR threshold check
                if valid_frame:
                    self.frames.append(frame)
            except (ValueError, IndexError) as e:
                print(f"Error parsing row {row_idx + 1}: {e}")

        return self.frames
