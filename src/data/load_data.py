import pandas as pd
import os

def load_raw_data(file_name, data_dir="data/raw"):
    """
    Load raw data from a CSV file.
    """
    file_path = os.path.join(data_dir, file_name)
    try:
        # Read the CSV file into a DataFrame
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except FileNotFoundError:
        # Handle missing file error
        print(f"Error: File not found at {file_path}")
        return None