import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import Tuple, Dict

class RiskEngine:
    """
    MNC-Grade Predictive Engine for Financial Tail-Risk.
    Uses XGBoost with SHAP for transparent risk classification.
    """
    
    def __init__(self, ticker_data: pd.DataFrame):
        self.raw_data = ticker_data
        self.model = None
        self.explainer = None
        self.feature_cols = []

    def engineer_features(self) -> pd.DataFrame:
        """Pipeline to create indicators and macro-proxies (VIX)."""
        df = self.raw_data.copy()
        
        # 1. Technical Indicators
        df['RSI'] = self._calc_rsi(df['Close'])
        df['Vol_Moving_Avg'] = df['Volume'].rolling(window=20).mean()
        df['Price_Volatility'] = df['High'] - df['Low']
        
        # 2. Temporal Lags (The 'Memory' of the model)
        for lag in [1, 3, 5]:
            df[f'Lag_Return_{lag}'] = df['Close'].pct_change(periods=lag)
        
        # 3. Target Variable: Tail Risk (Drop > 3% in next 5 days)
        # 1 = High Risk Event, 0 = Normal/Up
        df['Forward_Return'] = df['Close'].shift(-5).pct_change(periods=5)
        df['Target'] = (df['Forward_Return'] < -0.03).astype(int)
        
        # 4. Macro Proxy (VIX) - Fail-safe fetch
        try:
            vix = yf.download("^VIX", start=df.index[0], end=df.index[-1], progress=False)
            if not vix.empty and 'Close' in vix.columns:
                df['VIX_Close'] = vix['Close']
            else:
                df['VIX_Close'] = np.nan
        except Exception:
            df['VIX_Close'] = np.nan
            
        # Fill completely missing VIX and then drop other NaNs (from indicators/lags)
        if df['VIX_Close'].isna().all():
            df['VIX_Close'] = 0.0
            
        return df.dropna()

    def train_with_cv(self, df: pd.DataFrame) -> Dict:
        """Trains XGBoost using TimeSeriesSplit to prevent data leakage."""
        X = df.drop(columns=['Target', 'Forward_Return'])
        y = df['Target']
        self.feature_cols = X.columns.tolist()
        
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        # Calculate scale_pos_weight for class imbalance
        imbalance_ratio = (y == 0).sum() / (y == 1).sum()
        
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            scale_pos_weight=imbalance_ratio,
            objective='binary:logistic',
            tree_method='hist',
            base_score=0.5
        )

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            self.model.fit(X_train, y_train)
            
        # Final Metrics Template
        y_pred = self.model.predict(X)
        return {
            "Precision": precision_score(y, y_pred),
            "Recall": recall_score(y, y_pred),
            "F1": f1_score(y, y_pred)
        }

    def generate_explanations(self, X_sample: pd.DataFrame):
        """Generates SHAP values to explain the 'Why' behind a Risk Flag."""
        # Use native XGBoost SHAP calculation to bypass shap library parser bugs on v2.0+
        booster = self.model.get_booster()
        shap_values_with_bias = booster.predict(xgb.DMatrix(X_sample), pred_contribs=True)
        # XGBoost pred_contribs returns [features + 1] where the last column is the expected value (bias)
        shap_values = shap_values_with_bias[:, :-1]
        return shap_values

    def _calc_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))