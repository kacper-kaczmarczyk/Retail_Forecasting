import os
import zipfile
import numpy as np
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

# --- Configuration ---
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"
DATASET_NAME = "store-sales-time-series-forecasting"

def download_data():
    """
    Downloads the dataset from Kaggle API if it doesn't already exist.
    """
    if not os.path.exists(RAW_DATA_PATH):
        os.makedirs(RAW_DATA_PATH)
    
    # Check if data already exists to avoid re-downloading
    if os.path.exists(os.path.join(RAW_DATA_PATH, "train.csv")):
        print("✅ Data already exists. Skipping download.")
        return

    print(f"⬇️ Downloading data: {DATASET_NAME}...")
    try:
        api = KaggleApi()
        api.authenticate()
        api.competition_download_files(DATASET_NAME, path=RAW_DATA_PATH)
        
        # Unzip files
        zip_path = os.path.join(RAW_DATA_PATH, f"{DATASET_NAME}.zip")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DATA_PATH)
        print("✅ Data downloaded and extracted successfully.")
        
    except Exception as e:
        print(f"❌ Error during download: {e}")
        print("👉 Tip: Ensure 'kaggle.json' is in ~/.kaggle/ or %USERPROFILE%/.kaggle/")
        raise e

def create_toy_dataset():
    """
    Creates a simplified 'Toy Dataset' for immediate Power BI testing.
    It merges sales data with store metadata and adds dummy predictions.
    """
    print("⚙️ Processing data to create Toy Dataset...")
    
    try:
        # Load only necessary files
        df_train = pd.read_csv(os.path.join(RAW_DATA_PATH, "train.csv"))
        df_stores = pd.read_csv(os.path.join(RAW_DATA_PATH, "stores.csv"))
        
        # Convert date column
        df_train['date'] = pd.to_datetime(df_train['date'])
        
        # Merge: Sales + Store Metadata (Critical for Map visualizations in BI)
        # Note: The joining key in this dataset is 'store_nbr', not 'store_id'
        df_merged = df_train.merge(df_stores, on='store_nbr', how='left')
        
        # FILTERING FOR TOY DATA (Small subset for speed)
        # We take only the last 3 months and 5 specific stores
        latest_date = df_merged['date'].max()
        start_date = latest_date - pd.Timedelta(days=90)
        
        toy_df = df_merged[
            (df_merged['date'] >= start_date) & 
            (df_merged['store_nbr'].isin([1, 10, 20, 30, 40])) 
        ].copy()
        
        # Add "Mock" Predictions
        # Logic: Actual Sales + Random Noise (0.9 to 1.1 multiplier)
        np.random.seed(42)
        toy_df['predicted_sales'] = toy_df['sales'] * np.random.uniform(0.9, 1.1, len(toy_df))
        
        # Calculate Error for BI Analysis
        toy_df['error_diff'] = toy_df['sales'] - toy_df['predicted_sales']
        
        # Save to processed folder
        if not os.path.exists(PROCESSED_DATA_PATH):
            os.makedirs(PROCESSED_DATA_PATH)
            
        output_file = os.path.join(PROCESSED_DATA_PATH, "toy_data_bi_test.csv")
        toy_df.to_csv(output_file, index=False)
        
        print(f"🚀 Success! Toy data saved to: {output_file}")
        print(f"📊 Sample shape: {toy_df.shape}")
        print("💡 Next Step: Import this file into Power BI Desktop.")
        
    except FileNotFoundError:
        print("❌ Error: Raw files not found. Did the download finish successfully?")

if __name__ == "__main__":
    download_data()
    create_toy_dataset()