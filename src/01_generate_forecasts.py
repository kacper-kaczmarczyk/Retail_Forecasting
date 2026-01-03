import pandas as pd
import numpy as np
import lightgbm as lgb
import os

# --- Configuration ---
RAW_PATH = "data/raw"
PROCESSED_PATH = "data/processed"
OUTPUT_FILE = "final_forecasts.csv"

def generate_forecasts():
    print("🚀 Starting Granular Forecast Pipeline (Store + Family)...")
    
    # 1. Load Data
    print("⏳ Loading raw data...")
    df_train = pd.read_csv(os.path.join(RAW_PATH, "train.csv"))
    df_stores = pd.read_csv(os.path.join(RAW_PATH, "stores.csv"))
    
    df_train['date'] = pd.to_datetime(df_train['date'])
    
    # 2. Preprocessing & Feature Engineering
    print("🛠️ Feature Engineering (Granular Level)...")
    
    # Aggregating to Store + Family level
    df_grouped = df_train.groupby(['date', 'store_nbr', 'family'])[['sales', 'onpromotion']].sum().reset_index()
    
    # Date Features
    df_grouped['day_of_week'] = df_grouped['date'].dt.dayofweek
    df_grouped['day_of_month'] = df_grouped['date'].dt.day
    df_grouped['month'] = df_grouped['date'].dt.month
    
    # Lag Features (Must be calculated PER STORE AND FAMILY)
    print("   -> Calculating Lags (this might take a moment)...")
    # We create a temporary grouping key to speed up shifts
    df_grouped['group_key'] = df_grouped['store_nbr'].astype(str) + '_' + df_grouped['family']
    
    df_grouped['sales_lag_7'] = df_grouped.groupby('group_key')['sales'].shift(7)
    df_grouped['sales_lag_14'] = df_grouped.groupby('group_key')['sales'].shift(14)
    df_grouped['sales_lag_28'] = df_grouped.groupby('group_key')['sales'].shift(28)
    
    # Rolling Mean
    df_grouped['rolling_mean_7'] = df_grouped.groupby('group_key')['sales'].transform(lambda x: x.shift(1).rolling(7).mean())
    
    # Cleanup
    df_grouped = df_grouped.drop(columns=['group_key'])
    df_model = df_grouped.dropna().reset_index(drop=True)
    
    # 3. Train/Test Split (Simulation)
    test_days = 28
    latest_date = df_model['date'].max()
    cutoff_date = latest_date - pd.Timedelta(days=test_days)
    
    train_df = df_model[df_model['date'] < cutoff_date]
    test_df = df_model[df_model['date'] >= cutoff_date].copy()
    
    print(f"📊 Training Data: {train_df.shape[0]} rows")
    print(f"📊 Forecast Data: {test_df.shape[0]} rows")

    # 4. Training LightGBM
    features = [
        'store_nbr', 'family', 
        'day_of_week', 'day_of_month', 'month', 'onpromotion',
        'sales_lag_7', 'sales_lag_14', 'sales_lag_28', 'rolling_mean_7'
    ]
    target = 'sales'
    
    X_train = train_df[features].copy()
    y_train = train_df[target]
    X_test = test_df[features].copy()
    
    # Convert Categoricals
    for col in ['store_nbr', 'family']:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')
    
    print("🧠 Training Global LightGBM Model...")
    model = lgb.LGBMRegressor(
        objective='regression',
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # 5. Predictions
    print("🔮 Generating Predictions...")
    predictions = model.predict(X_test)
    predictions = np.maximum(predictions, 0)
    
    test_df['predicted_sales'] = predictions
    test_df['error_diff'] = test_df['sales'] - test_df['predicted_sales']
    
    # 6. Formatting for Power BI
    # Merge Store Metadata
    final_df = test_df.merge(df_stores, on='store_nbr', how='left')
    
    # Generate Unique ID
    final_df = final_df.reset_index(drop=True)
    final_df['id'] = final_df.index
    
    # Output Columns (Added 'family')
    output_columns = [
        'id', 'date', 'store_nbr', 'family', 'city', 'state', 'type', 'cluster',
        'onpromotion', 'sales', 'predicted_sales', 'error_diff'
    ]
    final_export = final_df[output_columns]
    
    # 7. Save
    if not os.path.exists(PROCESSED_PATH):
        os.makedirs(PROCESSED_PATH)
        
    output_path = os.path.join(PROCESSED_PATH, OUTPUT_FILE)
    final_export.to_csv(output_path, index=False)
    
    print(f"✅ DONE! Final forecasts saved to: {output_path}")
    print(f"📝 Columns: {list(final_export.columns)}")

if __name__ == "__main__":
    generate_forecasts()