import pandas as pd
import time

# Specify the file path (adjust for your setup)
file_path = r"\\wsl$\Ubuntu\root\.vs\Radar\out\build\linux-debug\Radar\OutputFile\detected_points.csv"

while True:
    try:
        # Read and process the file
        data = pd.read_csv(file_path)
        print(data)  # Print the contents of the file
    except Exception as e:
        print(f"Error reading file: {e}")

    # Wait for a short interval before reading again
    time.sleep(0.1)  # 10ms delay
