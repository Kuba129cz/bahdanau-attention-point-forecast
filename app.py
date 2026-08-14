import json
import pathlib
from datetime import date
from types import SimpleNamespace

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from meteo import pipeline
from predict import EnsemblePredictor

LATITUDE = 49.13114
LONGITUDE = 15.18067

@st.cache_data
def load_config_cached(config_file_str: str) -> SimpleNamespace:
    """Loads config.json from disk only on the first call."""
    config_file = pathlib.Path(config_file_str)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_file}")

    with open(config_file, mode="r", encoding="utf-8") as f:
        config_dict = json.load(f)

    return SimpleNamespace(**config_dict)

@st.cache_resource
def get_predictor() -> EnsemblePredictor:
    """Initializes and caches the EnsemblePredictor."""
    return EnsemblePredictor()

def main():
    st.title("Bahdanau Attention PV Point Forecast")
    
    args = load_config_cached("config.json")

    with st.container(border=True):
        selected_day = st.date_input(
            label="Select day",
            value=date.today(),
            max_value=date.today(),
            format="YYYY/MM/DD"
        )
        
        if st.button("Predict"):
            st.info(f"Running prediction for day: {selected_day}")
            
            try:
                # Prepare Data
                with st.spinner("Preparing history dataset..."):
                    df_history, df_future, df_today_energy, future_times = pipeline.prepare_history_dataset(
                        predicted_day=selected_day,
                        lookback_cols=args.lookback_cols,
                        horizon_cols=args.horizon_cols
                    )
                
                # 2. Run Inference
                with st.spinner("Running ensemble models..."):
                    predictor = get_predictor()
                    predictions = predictor.predict(
                        df_history=df_history,
                        df_future=df_future,
                        df_time=future_times,
                        known_targets_df=df_today_energy
                    )

                st.success("Prediction completed successfully!")

                # ==========================================
                # Process Data for Tables
                # ==========================================
                chart_data = []
                for p in predictions:
                    chart_data.append({
                        "Time": pd.to_datetime(p["time"]),
                        "Max Power": max(p["model_predictions"]),
                        "Average Power": p["power"],
                        "Min Power": min(p["model_predictions"]),
                    })
                df_chart = pd.DataFrame(chart_data).set_index("Time")

                models_data = []
                for p in predictions:
                    row = {"Time": pd.to_datetime(p["time"]).strftime("%H:%M")}
                    for i, m_pred in enumerate(p["model_predictions"]):
                        row[f"Model {i+1}"] = m_pred
                    models_data.append(row)
                df_models = pd.DataFrame(models_data).set_index("Time")

                # ==========================================
                # Render UI elements
                # ==========================================
                tab_chart, tab_models, tab_table, tab_hist, tab_fut, tab_energy = st.tabs([
                    "📈 Chart",
                    "📊 Individual Models",
                    "📋 Predictions Data",
                    "🌤️ Meteo previous day", 
                    "🌤️ Meteo for predicted day", 
                    "⚡ Target Energy"
                ])

                with tab_chart:
                    st.subheader("Ensemble Power Prediction (kWh)")
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=df_chart.index, y=df_chart["Max Power"], 
                        mode="lines", name="Max Predicted Power", 
                        line=dict(width=0), 
                        showlegend=False,
                        hovertemplate="Max Predicted Power: %{y:.2f} kWh<extra></extra>"
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=df_chart.index, y=df_chart["Min Power"], 
                        mode="lines", name="Min Predicted Power", 
                        line=dict(width=0),
                        fill="tonexty", 
                        fillcolor="rgba(255, 127, 14, 0.2)",
                        showlegend=False,
                        hovertemplate="Min Predicted Power: %{y:.2f} kWh<extra></extra>"
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=df_chart.index, y=df_chart["Average Power"], 
                        mode="lines+markers", name="Predicted Power", 
                        line=dict(color="rgba(255, 127, 14, 1)", width=3),
                        hovertemplate="Predicted Power: %{y:.2f} kWh<extra></extra>"
                    ))
                    
                    if df_today_energy is not None and not df_today_energy.empty:
                        df_actuals = df_today_energy.copy()
                        
                        target_col = getattr(args, 'target_col', df_actuals.columns[0])
                        if target_col not in df_actuals.columns:
                            target_col = df_actuals.columns[0]
                        
                        if isinstance(df_actuals.index, pd.DatetimeIndex):
                            x_actuals = df_actuals.index
                        else:
                            time_cols = [c for c in df_actuals.columns if c.lower() in ['time', 'timestamp', 'date']]
                            if time_cols:
                                x_actuals = pd.to_datetime(df_actuals[time_cols[0]])
                            else:
                                x_actuals = df_chart.index[:len(df_actuals)]
                                
                        fig.add_trace(go.Scatter(
                            x=x_actuals, y=df_actuals[target_col], 
                            mode="lines+markers", name="Actual Power", 
                            line=dict(color="#d62728", dash="dash", width=2)
                        ))

                    fig.update_layout(
                        xaxis_title="Time",
                        yaxis_title="Power (kWh)",
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    
                    st.plotly_chart(fig, width="stretch")

                with tab_models:
                    st.subheader("Predictions by Individual Models")
                    st.dataframe(df_models, width="stretch")

                with tab_table:
                    st.subheader("Tabular Predictions")
                    st.dataframe(predictions, width="stretch")

                with tab_hist:
                    st.subheader("Historical Data (df_history)")
                    st.dataframe(df_history, width="stretch")

                with tab_fut:
                    st.subheader("Forecast Horizon Data (df_future)")
                    st.dataframe(df_future, width="stretch")

                with tab_energy:
                    st.subheader("Target Variable (df_today_energy)")
                    st.dataframe(df_today_energy, width="stretch")

            except requests.exceptions.HTTPError as e:
                st.error(f"Failed to fetch data from the API. Please check your credentials or try again later. Details: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred while processing data: {e}")

if __name__ == "__main__":
    main()