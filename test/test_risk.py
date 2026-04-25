import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.utils.data_loader import fetch_historical_data, TickerNotFoundError
from src.models.risk_engine import RiskEngine

class TestRiskArchitect(unittest.TestCase):

    def setUp(self):
        """Setup mock data for testing."""
        self.mock_data = pd.DataFrame({
            'Close': [150, 155, 152, 148, 145] * 10,
            'High': [152, 157, 154, 150, 147] * 10,
            'Low': [148, 153, 150, 146, 143] * 10,
            'Volume': [1000, 1100, 1050, 900, 850] * 10,
            'SMA_20': [150] * 50,
            'SMA_50': [150] * 50
        })
        self.engine = RiskEngine(self.mock_data)

    def test_feature_engineering_output(self):
        """Test if technical indicators are calculated correctly."""
        df_processed = self.engine.engineer_features()
        self.assertIn('RSI', df_processed.columns)
        self.assertIn('VIX_Close', df_processed.columns)
        self.assertFalse(df_processed.isnull().values.any())

    @patch('src.utils.data_loader.yf.Ticker')
    def test_data_ingestion_resilience(self, mock_ticker):
        """Test how the system handles a failed ticker search."""
        # Simulate empty info meaning TickerNotFoundError
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {}
        mock_ticker.return_value = mock_ticker_instance
        
        with self.assertRaises(TickerNotFoundError):
            fetch_historical_data("INVALID_TICKER")

    def test_model_prediction_logic(self):
        """Ensure the XGBoost model outputs a binary classification."""
        # Simple dummy train for logic check
        df = self.engine.engineer_features()
        self.engine.train_with_cv(df)
        X_sample = df.drop(columns=['Target', 'Forward_Return']).iloc[-1:]
        prediction = self.engine.model.predict(X_sample)
        self.assertIn(prediction[0], [0, 1])

if __name__ == '__main__':
    unittest.main()
