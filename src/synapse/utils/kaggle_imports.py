import os, glob
import pandas as pd
import kagglehub

def kaggle_download_csv(dataset_name: str) -> pd.DataFrame:
    root = kagglehub.dataset_download(dataset_name)
    csv_files = glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
    assert csv_files, f"No CSV files found under: {root}"
    csv_path = csv_files[0]
    return pd.read_csv(csv_path, header=None)