import os
import sys
import traceback
from dotenv import load_dotenv

# Load env before importing
load_dotenv()

from src.utils.data_loader import fetch_ticker_with_news
from src.models.risk_engine import RiskEngine
from src.agents.orchestrator import RiskOrchestrator

def run_chaotic_ticker():
    print("\n" + "="*60)
    print("SIMULATION 1: The 'Chaotic Ticker' Run (TSLA)")
    print("="*60)
    ticker = "TSLA"
    try:
        data_payload = fetch_ticker_with_news(ticker, news_page_size=10)
        hist_df = data_payload['history']
        news = data_payload['news']
        headlines = [n['title'] for n in news]
        print(f"Fetched {len(headlines)} headlines for {ticker}.")

        engine = RiskEngine(hist_df)
        engineered_df = engine.engineer_features()
        metrics = engine.train_with_cv(engineered_df)
        
        latest_data = engineered_df.iloc[[-1]].drop(columns=['Target', 'Forward_Return'])
        shap_explanations = engine.generate_explanations(latest_data)
        print("ML Engine Trained & SHAP Explanations Generated.")
        
        model_data_summary = {
            "metrics": metrics,
            "shap_values": shap_explanations
        }
        
        print("Triggering Agents...")
        orchestrator = RiskOrchestrator(model_data=model_data_summary, news_headlines=headlines)
        final_report = orchestrator.run_sprint()
        
        if hasattr(final_report, 'raw'):
            print("\nFinal Report (JSON):\n", final_report.raw)
        else:
            print("\nFinal Report (JSON):\n", str(final_report))

    except Exception as e:
        print(f"Failed during Chaotic Ticker Run:\n{e}")
        traceback.print_exc()

def run_news_drought():
    print("\n" + "="*60)
    print("SIMULATION 2: The 'News Drought' Simulation")
    print("="*60)
    
    # Temporarily remove NEWS_API_KEY
    original_key = os.environ.get("NEWS_API_KEY")
    if "NEWS_API_KEY" in os.environ:
        del os.environ["NEWS_API_KEY"]
        
    try:
        print("Attempting to run analysis without NEWS_API_KEY...")
        
        # This should fail or return empty news
        data_payload = fetch_ticker_with_news("AAPL", news_page_size=5)
        news = data_payload.get('news', [])
        headlines = [n['title'] for n in news]
        
        print(f"Headlines fetched: {len(headlines)}")
        
        model_data_summary = {"dummy": "data"}
        
        print("Triggering Orchestrator...")
        # This should trigger the ValueError
        orchestrator = RiskOrchestrator(model_data=model_data_summary, news_headlines=headlines)
        print("Wait, orchestrator did NOT fail. This is unexpected.")
        
    except Exception as e:
        print(f"Success! System Failed Loudly with Error:\n---> {type(e).__name__}: {e}")
    finally:
        if original_key:
            os.environ["NEWS_API_KEY"] = original_key

if __name__ == "__main__":
    run_news_drought()
    run_chaotic_ticker()
