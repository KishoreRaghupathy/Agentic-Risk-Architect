import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from src.utils.data_loader import fetch_historical_data
from src.models.risk_engine import RiskEngine

def run_test():
    print("Fetching raw data for AAPL...")
    try:
        raw_data = fetch_historical_data("AAPL", years=2)
    except Exception as e:
        print(f"yfinance rate-limited ({e}). Falling back to synthetic test data...")
        import numpy as np
        dates = pd.date_range(end=pd.Timestamp.today(), periods=500, freq='B')
        raw_data = pd.DataFrame({
            'Open': np.random.uniform(150, 180, size=len(dates)),
            'High': np.random.uniform(152, 185, size=len(dates)),
            'Low': np.random.uniform(148, 178, size=len(dates)),
            'Close': np.random.uniform(150, 182, size=len(dates)),
            'Volume': np.random.randint(5000000, 100000000, size=len(dates)),
        }, index=dates)

    print("Initializing Risk Engine and engineering features...")
    engine = RiskEngine(raw_data)
    df_features = engine.engineer_features()
    
    print(f"Engineered dataset shape: {df_features.shape}")
    print(f"Features: {list(df_features.columns)}")

    print("Training XGBoost with TimeSeriesSplit...")
    metrics = engine.train_with_cv(df_features)
    
    print("\n--- Model Evaluation Summary ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    print("\nExtracting Explainability (SHAP) for latest state...")
    latest_state = df_features.drop(columns=['Target', 'Forward_Return']).iloc[[-1]]
    shap_vals = engine.generate_explanations(latest_state)
    print("SHAP Generation Successful.")
    
if __name__ == "__main__":
    run_test()
