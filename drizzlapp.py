import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

from visualizations import rainfall_gauge
from visualizations import forecast_comparison
from visualizations import probability_meter
from visualizations import regime_probability_chart

from risk import calculate_risk_score
from risk import get_risk_level
from risk import get_risk_message

from verification import compute_all_metrics, print_metrics_report
from historical_replay import historical_replay, performance_summary, load_performance_log
from district_forecast import forecast_all_districts, print_district_summary, create_district_geojson, load_district_forecast
from real_data import build_real_dataset, DISTRICT_COORDS

import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="🌧️ DRIZZL", layout="wide", initial_sidebar_state="expanded")

classifier = joblib.load("models/regime_classifier.pkl")
active_model = joblib.load("models/active_model.pkl")
break_model = joblib.load("models/break_model.pkl")
depression_model = joblib.load("models/depression_model.pkl")

st.title("🌧️ DRIZZL")
st.caption("Regime-Aware AI Post-Processing of Monsoon Rainfall Forecasts")

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation",
    ["🔮 Single Forecast", "🗺️ District Forecast", "📊 Verification Metrics", "📈 Historical Replay", "📥 Data Integration"],
    index=0
)

# ============================================================
# PAGE 1: SINGLE FORECAST (Original)
# ============================================================
if page == "🔮 Single Forecast":
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Temperature (°C)", 20, 40, 30)
        humidity = st.slider("Humidity (%)", 40, 100, 80)
        wind = st.slider("Wind Speed (km/h)", 0, 40, 10)
    with col2:
        pressure = st.slider("Pressure (hPa)", 970, 1020, 1000)
        nwp = st.slider("NWP Rainfall Forecast (mm)", 0, 250, 50)

    if st.button("Predict", type="primary", use_container_width=True):
        regime_input = pd.DataFrame(
            [[temperature, humidity, wind, pressure]],
            columns=["temperature", "humidity", "wind", "pressure"]
        )

        regime = classifier.predict(regime_input)[0]
        probabilities = classifier.predict_proba(regime_input)[0]
        confidence = max(probabilities) * 100
        reliability = min(100, confidence * 0.9)
        
        st.subheader("Weather Regime")
        st.success(regime)
        st.metric("Regime Confidence", f"{confidence:.1f}%")
        st.metric("Forecast Reliability", f"{reliability:.0f}/100")
        
        st.subheader("🌦 Regime Analysis")
        regime_probability_chart(probabilities, classifier.classes_)
        for label, prob in zip(classifier.classes_, probabilities):
            st.write(f"**{label}** : {prob*100:.1f}%")

        full_input = pd.DataFrame(
            [[temperature, humidity, wind, pressure, nwp]],
            columns=["temperature", "humidity", "wind", "pressure", "nwp_rainfall"]
        )

        if regime == "Active":
            corrected = active_model.predict(full_input)[0]
        elif regime == "Break":
            corrected = break_model.predict(full_input)[0]
        else:
            corrected = depression_model.predict(full_input)[0]

        risk_score = calculate_risk_score(corrected, confidence, reliability)
        risk_level = get_risk_level(risk_score)
        risk_message = get_risk_message(risk_score)

        st.subheader("Forecast Comparison")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Raw NWP", f"{nwp:.1f} mm")
        with c2:
            st.metric("DRIZZL Forecast", f"{corrected:.1f} mm")

        if corrected < 15:
            category = "🌦 Light Rain"
        elif corrected < 65:
            category = "🌧 Moderate Rain"
        elif corrected < 115:
            category = "🌧 Heavy Rain"
        elif corrected < 205:
            category = "⛈ Very Heavy Rain"
        else:
            category = "🚨 Extreme Rainfall"
        st.metric("IMD Rainfall Category", category)

        st.subheader("Why DRIZZL Changed The Forecast")
        if humidity > 80:
            st.write(" High humidity increased rainfall potential")
        if wind > 25:
            st.write(" Strong winds support moisture transport")
        if pressure < 995:
            st.write(" Low pressure favors rainfall development")
        if regime == "Depression":
            st.write(" Depression regime detected")
        if regime == "Active":
            st.write(" Active monsoon regime detected")

        correction = corrected - nwp
        if correction > 0:
            st.info(f"DRIZZL increased the forecast by {correction:.1f} mm")
        else:
            st.info(f"DRIZZL reduced the forecast by {abs(correction):.1f} mm")

        forecast_comparison(nwp, corrected)
        rainfall_gauge(corrected)
        probability = min(100, corrected / 2)
        probability_meter(probability)

        st.subheader("🚨 Risk Intelligence")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Risk Score", f"{risk_score}/100")
        with c2:
            st.metric("Risk Level", risk_level)
        st.warning(risk_message)

# ============================================================
# PAGE 2: DISTRICT FORECAST
# ============================================================
elif page == "🗺️ District Forecast":
    st.subheader("🗺️ District-Level Rainfall Forecast")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Generate Forecast for All Districts", type="primary", use_container_width=True):
            with st.spinner("Generating district forecasts..."):
                forecasts = forecast_all_districts()
                st.session_state['district_forecasts'] = forecasts
                st.success(f"Generated forecasts for {len(forecasts)} districts")
    
    if 'district_forecasts' in st.session_state:
        forecasts = st.session_state['district_forecasts']
        valid = [f for f in forecasts if "error" not in f]
        
        # Summary table
        st.subheader("📋 District Forecast Summary")
        
        df_display = pd.DataFrame([{
            "District": f["district"],
            "State": f["state"],
            "NWP (mm)": round(f["nwp_rainfall"], 1),
            "DRIZZL (mm)": round(f["corrected_rainfall"], 1),
            "Regime": f["regime"],
            "Confidence (%)": round(f["regime_confidence"], 1),
            "IMD Category": f"{f['imd_category_emoji']} {f['imd_category']}",
            "Risk Level": f["risk_level"],
            "Risk Score": f["risk_score"]
        } for f in valid])
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        # Map visualization
        st.subheader("🗺️ Spatial Forecast Map")
        geojson = create_district_geojson(forecasts)
        
        # Create map using plotly (using scatter_map for newer plotly versions)
        map_df = pd.DataFrame([{
            "lat": f["latitude"],
            "lon": f["longitude"],
            "district": f["district"],
            "rainfall": f["corrected_rainfall"],
            "regime": f["regime"],
            "category": f["imd_category"],
            "risk": f["risk_level"]
        } for f in valid])
        
        fig = px.scatter_map(
            map_df, lat="lat", lon="lon", 
            color="rainfall", size="rainfall",
            hover_data=["district", "regime", "category", "risk"],
            color_continuous_scale=["lightgreen", "gold", "orange", "red", "darkred"],
            size_max=30, zoom=3.5,
            map_style="carto-positron",
            title="DRIZZL Corrected Rainfall Forecast by District"
        )
        fig.update_layout(height=600, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
        
        # Regime distribution
        st.subheader("📊 Regime Distribution")
        regime_counts = pd.Series([f["regime"] for f in valid]).value_counts()
        fig_pie = px.pie(values=regime_counts.values, names=regime_counts.index, 
                         title="Districts by Weather Regime", hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Download
        st.download_button(
            "📥 Download Forecast (JSON)",
            data=json.dumps(forecasts, indent=2, default=str),
            file_name=f"district_forecast_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json"
        )

# ============================================================
# PAGE 3: VERIFICATION METRICS
# ============================================================
elif page == "📊 Verification Metrics":
    st.subheader("📊 Forecast Verification Metrics")
    st.caption("RMSE, MAE, Bias, Correlation | POD, FAR, CSI, ETS, HSS, FSS")
    
    # Load validation data
    try:
        val_df = pd.read_csv("data/weather_data.csv")
        val_df["date"] = pd.date_range("2023-06-01", periods=len(val_df), freq="D").astype(str)
        
        # Run predictions
        predictions = []
        actuals = []
        
        for _, row in val_df.iterrows():
            X_regime = pd.DataFrame([[
                row["temperature"], row["humidity"], row["wind"], row["pressure"]
            ]], columns=["temperature", "humidity", "wind", "pressure"])
            
            X_full = pd.DataFrame([[
                row["temperature"], row["humidity"], row["wind"], 
                row["pressure"], row["nwp_rainfall"]
            ]], columns=["temperature", "humidity", "wind", "pressure", "nwp_rainfall"])
            
            regime = classifier.predict(X_regime)[0]
            probs = classifier.predict_proba(X_regime)[0]
            confidence = max(probs) * 100
            reliability = min(100, confidence * 0.9)
            
            if regime == "Active":
                corrected = active_model.predict(X_full)[0]
            elif regime == "Break":
                corrected = break_model.predict(X_full)[0]
            else:
                corrected = depression_model.predict(X_full)[0]
            
            predictions.append(corrected)
            actuals.append(row["actual_rainfall"])
        
        y_true = np.array(actuals)
        y_pred = np.array(predictions)
        
        # Compute metrics
        metrics = compute_all_metrics(y_true, y_pred)
        
        # Continuous metrics
        st.subheader("📈 Continuous Metrics")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("RMSE", f"{metrics['continuous']['RMSE']:.2f} mm")
        with c2:
            st.metric("MAE", f"{metrics['continuous']['MAE']:.2f} mm")
        with c3:
            st.metric("Bias", f"{metrics['continuous']['Bias']:.2f} mm")
        with c4:
            st.metric("Correlation", f"{metrics['continuous']['Correlation']:.3f}")
        
        # Scatter plot
        fig_scatter = px.scatter(x=y_true, y=y_pred, 
                                 labels={"x": "Actual Rainfall (mm)", "y": "Predicted Rainfall (mm)"},
                                 title="Predicted vs Actual Rainfall",
                                 opacity=0.6)
        fig_scatter.add_trace(go.Scatter(x=[0, 250], y=[0, 250], mode='lines', 
                                         line=dict(color='red', dash='dash'), name='Perfect'))
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Categorical metrics
        st.subheader("📋 Categorical Metrics (IMD Thresholds)")
        
        thresholds = [2.5, 15, 65, 115]
        threshold_names = ["Light (2.5mm)", "Moderate (15mm)", "Heavy (65mm)", "Very Heavy (115mm)"]
        
        cat_df = pd.DataFrame()
        for thresh, name in zip(thresholds, threshold_names):
            cat_df[name] = {
                "POD": metrics["categorical"][f"threshold_{thresh}mm"]["POD"],
                "FAR": metrics["categorical"][f"threshold_{thresh}mm"]["FAR"],
                "CSI": metrics["categorical"][f"threshold_{thresh}mm"]["CSI"],
                "ETS": metrics["categorical"][f"threshold_{thresh}mm"]["ETS"],
                "HSS": metrics["categorical"][f"threshold_{thresh}mm"]["HSS"],
                "FSS": metrics["categorical"][f"threshold_{thresh}mm"]["FSS"],
            }
        
        cat_df = cat_df.T
        st.dataframe(cat_df.style.format("{:.3f}").background_gradient(cmap="RdYlGn"), 
                     use_container_width=True)
        
        # Metric explanations
        with st.expander("📖 Metric Definitions"):
            st.markdown("""
            **Continuous Metrics:**
            - **RMSE**: Root Mean Square Error - overall prediction error
            - **MAE**: Mean Absolute Error - average absolute error
            - **Bias**: Mean Bias Error - positive = overprediction
            - **Correlation**: Pearson correlation between predicted and actual
            
            **Categorical Metrics (for rainfall ≥ threshold):**
            - **POD (Probability of Detection)**: Hits / (Hits + Misses) - fraction of events detected
            - **FAR (False Alarm Ratio)**: False Alarms / (Hits + False Alarms) - fraction of false alarms
            - **CSI (Critical Success Index)**: Hits / (Hits + Misses + False Alarms) - threat score
            - **ETS (Equitable Threat Score)**: CSI adjusted for random hits
            - **HSS (Heidke Skill Score)**: Accuracy relative to random chance
            - **FSS (Fractional Skill Score)**: Spatial/temporal neighborhood verification
            """)
            
    except Exception as e:
        st.error(f"Error computing metrics: {e}")

# ============================================================
# PAGE 4: HISTORICAL REPLAY
# ============================================================
elif page == "📈 Historical Replay":
    st.subheader("📈 Historical Performance Replay")
    st.caption("Time-series validation: model performance over the monsoon season")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Run Historical Replay", type="primary", use_container_width=True):
            with st.spinner("Running historical replay on full dataset..."):
                try:
                    df = pd.read_csv("data/weather_data.csv")
                    df["date"] = pd.date_range("2023-06-01", periods=len(df), freq="D").astype(str)
                    
                    models = {
                        "classifier": classifier,
                        "active": active_model,
                        "break": break_model,
                        "depression": depression_model
                    }
                    
                    results = historical_replay(df, models)
                    summary = performance_summary(results)
                    
                    st.session_state['replay_results'] = results
                    st.session_state['replay_summary'] = summary
                    st.success(f"Replay complete: {len(results)} days analyzed")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    if 'replay_results' in st.session_state:
        results = st.session_state['replay_results']
        summary = st.session_state['replay_summary']
        
        # Summary metrics
        st.subheader("📊 Performance Summary")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Mean RMSE", f"{summary['continuous']['RMSE']['mean']:.2f} mm")
        with c2:
            st.metric("Mean MAE", f"{summary['continuous']['MAE']['mean']:.2f} mm")
        with c3:
            st.metric("Mean Bias", f"{summary['continuous']['Bias']['mean']:.2f} mm")
        with c4:
            corr_mean = summary['continuous'].get('Correlation', {}).get('mean', np.nan)
            st.metric("Mean Correlation", f"{corr_mean:.3f}" if not np.isnan(corr_mean) else "N/A")
        
        # Time series plots
        dates = [d["date"] for d in results]
        
        # RMSE over time
        fig_rmse = px.line(x=dates, y=[d["continuous"]["RMSE"] for d in results],
                           labels={"x": "Date", "y": "RMSE (mm)"},
                           title="RMSE Over Time")
        fig_rmse.update_layout(height=350)
        st.plotly_chart(fig_rmse, use_container_width=True)
        
        # Categorical metrics over time
        csi_vals = [d["categorical"]["threshold_15mm"]["CSI"] for d in results]
        pod_vals = [d["categorical"]["threshold_15mm"]["POD"] for d in results]
        far_vals = [d["categorical"]["threshold_15mm"]["FAR"] for d in results]
        
        fig_cat = go.Figure()
        fig_cat.add_trace(go.Scatter(x=dates, y=csi_vals, name="CSI", mode='lines'))
        fig_cat.add_trace(go.Scatter(x=dates, y=pod_vals, name="POD", mode='lines'))
        fig_cat.add_trace(go.Scatter(x=dates, y=far_vals, name="FAR", mode='lines'))
        fig_cat.update_layout(title="Categorical Metrics (15mm threshold) Over Time",
                              xaxis_title="Date", yaxis_title="Score", height=350)
        st.plotly_chart(fig_cat, use_container_width=True)
        
        # Predicted vs Actual over time
        fig_compare = go.Figure()
        fig_compare.add_trace(go.Scatter(x=dates, y=[d["mean_predicted"] for d in results],
                                         name="Mean Predicted", mode='lines'))
        fig_compare.add_trace(go.Scatter(x=dates, y=[d["mean_actual"] for d in results],
                                         name="Mean Actual", mode='lines'))
        fig_compare.update_layout(title="Mean Daily Rainfall: Predicted vs Actual",
                                  xaxis_title="Date", yaxis_title="Rainfall (mm)", height=350)
        st.plotly_chart(fig_compare, use_container_width=True)
        
        # Bias over time
        fig_bias = px.line(x=dates, y=[d["continuous"]["Bias"] for d in results],
                           labels={"x": "Date", "y": "Bias (mm)"},
                           title="Bias Over Time (Positive = Overprediction)")
        fig_bias.add_hline(y=0, line_dash="dash", line_color="red")
        fig_bias.update_layout(height=300)
        st.plotly_chart(fig_bias, use_container_width=True)

# ============================================================
# PAGE 5: DATA INTEGRATION
# ============================================================
elif page == "📥 Data Integration":
    st.subheader("📥 Real Weather Dataset Integration")
    st.caption("IMD Gridded Rainfall | ERA5 Reanalysis | NWP Forecasts | District-Level Products")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Available Data Sources")
        st.markdown("""
        | Source | Variables | Resolution | Status |
        |--------|-----------|------------|--------|
        | **IMD Gridded Rainfall** | Daily rainfall | 0.25° × 0.25° | 🟡 Demo |
        | **ERA5 Reanalysis** | T, RH, Wind, Pressure | 0.25° × 0.25° | 🟡 Demo |
        | **NWP Forecast** | Rainfall forecast | Model grid | 🟡 Demo |
        | **District Coordinates** | 15 major districts | Point locations | ✅ Ready |
        """)
    
    with col2:
        st.markdown("### Generate Training Dataset")
        n_districts = st.selectbox("Districts", [5, 10, 15], index=2)
        n_days = st.selectbox("Days", [30, 60, 90, 120], index=3)
        
        if st.button("Build Dataset", type="primary", use_container_width=True):
            with st.spinner("Building dataset from weather sources..."):
                districts = list(DISTRICT_COORDS.keys())[:n_districts]
                end_date = datetime(2023, 6, 1) + timedelta(days=n_days)
                df = build_real_dataset(districts=districts, 
                                       start_date=datetime(2023, 6, 1),
                                       end_date=end_date)
                df.to_csv("data/real_weather_data.csv", index=False)
                st.success(f"Generated {len(df)} records for {n_districts} districts over {n_days} days")
                st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown("### District Coverage")
    district_df = pd.DataFrame([
        {"District": k, "State": v["state"], "Lat": v["lat"], "Lon": v["lon"]}
        for k, v in DISTRICT_COORDS.items()
    ])
    
    fig_map = px.scatter_map(
        district_df, lat="Lat", lon="Lon",
        hover_data=["District", "State"],
        color="State", size_max=15, zoom=3.5,
        map_style="carto-positron",
        title="DRIZZL District Coverage"
    )
    fig_map.update_layout(height=500, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_map, use_container_width=True)
    
    st.markdown("### Data Pipeline Architecture")
    st.code("""
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │  IMD Rain   │    │   ERA5      │    │   NWP       │
    │  (Observed) │    │  (Reanalysis)    │  (Forecast) │
    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
           │                  │                  │
           └──────────────────┼──────────────────┘
                              ▼
                    ┌───────────────────┐
                    │  Data Fusion &    │
                    │  Quality Control  │
                    └─────────┬─────────┘
                              ▼
                    ┌───────────────────┐
                    │  Regime-Aware     │
                    │  ML Training      │
                    └─────────┬─────────┘
                              ▼
                    ┌───────────────────┐
                    │  Verification &   │
                    │  Historical Replay│
                    └───────────────────┘
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("DRIZZL v1.0 | Smart India Hackathon")
st.sidebar.caption("Regime-Aware AI Post-Processing")