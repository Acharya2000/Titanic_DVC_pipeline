import pandas as pd
import numpy as np
import os 

class DataIngestion:
    def __init__(self, source_path, raw_dir="data/raw", file_name="train.csv"):
        """
        source_path: str - Path to the original dataset (downloaded from Kaggle)
        raw_dir: str - Directory where raw data should be stored
        file_name: str - File name to save inside raw_dir
        """
        self.source_path = source_path
        self.raw_dir = raw_dir
        self.file_name = file_name
        self.raw_file_path = os.path.join(self.raw_dir, self.file_name)

        # Ensure raw directory exists
        os.makedirs(self.raw_dir, exist_ok=True)

    def load_data(self):
        """Loads CSV from source_path into a pandas DataFrame"""
        df = pd.read_csv(self.source_path)
        return df

    def save_raw(self):
        """Loads the data and saves a copy into data/raw/ folder"""
        df = self.load_data()
        df.to_csv(self.raw_file_path, index=False)
        print(f"[INFO] Raw data saved at: {self.raw_file_path}")
        return self.raw_file_path
    



