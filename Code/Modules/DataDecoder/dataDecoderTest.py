import dataDecoder
import os

script_dir = os.path.dirname(os.path.abspath(__file__))  # Current script directory
log_file = os.path.join(script_dir, '3_Targer_StraightLine_attempt3_log_2024-12-09.csv')
frames = dataDecoder.decodeData(log_file)