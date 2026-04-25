import streamlit as st
import pandas as pd
import time
import os
from dotenv import load_dotenv

from src.utils.data_loader import fetch_ticker_with_news
from src.models.risk_engine import RiskEngine
from src.agents.orchestrator import RiskOrchestrator

# Load environment variables (force override to catch changes while server is running)
load_dotenv(override=True)

st.set_page_config(page_title="Agentic Risk Architect", layout="wide", page_icon="📈")

st.title("🛡️ Agentic Risk Architect")
st.markdown("Enterprise-grade ML & Multi-Agent Risk Advisory Platform")

# Sidebar for inputs
with st.sidebar:
    st.header("Parameters")
    ticker = st.text_input("Ticker Symbol", value="AAPL")
    analyze_btn = st.button("Run Risk Analysis")

if analyze_btn:
    if not os.getenv("NEWS_API_KEY"):
        st.error("Please set NEWS_API_KEY in the .env file.")
        st.stop()
    if not (os.getenv("GOOGLE_API_KEY") and os.getenv("GROQ_API_KEY")):
        st.error("Please set both GOOGLE_API_KEY and GROQ_API_KEY in the .env file.")
        st.stop()

    st.info(f"Initiating workflow for {ticker}...")

    # Phase 1: Data Ingestion
    with st.spinner("Ingesting Market Data & News..."):
        start_time = time.time()
        try:
            data_payload = fetch_ticker_with_news(ticker)
        except Exception as e:
            st.error(f"Data ingestion failed: {e}")
            st.stop()
        
        hist_df = data_payload['history']
        news = data_payload['news']
        headlines = [n['title'] for n in news]
        ingestion_time = time.time() - start_time

    # Phase 2: ML Engine
    with st.spinner("Training XGBoost Risk Model & SHAP analysis..."):
        start_time = time.time()
        engine = RiskEngine(hist_df)
        engineered_df = engine.engineer_features()
        metrics = engine.train_with_cv(engineered_df)
        
        # Get latest data point for inference
        latest_data = engineered_df.iloc[[-1]].drop(columns=['Target', 'Forward_Return'])
        shap_explanations = engine.generate_explanations(latest_data)
        ml_time = time.time() - start_time

    # Phase 3: Agent Orchestration
    with st.spinner("CrewAI Agents Deliberating..."):
        start_time = time.time()
        
        model_data_summary = {
            "metrics": metrics,
            "shap_values": shap_explanations
        }
        
        orchestrator = RiskOrchestrator(model_data=model_data_summary, news_headlines=headlines)
        final_report = orchestrator.run_sprint()
        agent_time = time.time() - start_time

    st.success("Analysis Complete!")

    # Display Results
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Final Executive Risk Advisory")
        # Ensure it works whether final_report is a pydantic object, a string, or CrewOutput
        if hasattr(final_report, 'raw'):
            report_content = final_report.raw
        else:
            report_content = str(final_report)
            
        st.markdown(f"```json\n{report_content}\n```")

    with col2:
        st.subheader("System Health Report")
        st.metric("Ingestion Latency", f"{ingestion_time:.2f}s")
        st.metric("Model Training Latency", f"{ml_time:.2f}s")
        st.metric("Agent Response Time", f"{agent_time:.2f}s")
        st.metric("Total Processing Time", f"{(ingestion_time + ml_time + agent_time):.2f}s")

        st.subheader("ML Model Metrics")
        st.json(metrics)
