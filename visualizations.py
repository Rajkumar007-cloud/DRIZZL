import streamlit as st
import plotly.graph_objects as go


def rainfall_gauge(corrected):

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=corrected,
            title={"text": "🌧️ DRIZZL Rainfall Forecast (mm)"},
            gauge={
                "axis": {"range": [0, 250]},
                "bar": {"color": "cyan"},
                "steps": [
                    {"range": [0, 64], "color": "lightgreen"},
                    {"range": [64, 115], "color": "gold"},
                    {"range": [115, 250], "color": "tomato"}
                ]
            }
        )
    )

    fig.update_layout(height=400)
    st.plotly_chart(fig,width="stretch")

def forecast_comparison(nwp, corrected):
    correction = corrected - nwp
    st.metric(
        "Forecast Improvement",
        f"{abs(correction):.1f} mm"
    )
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=["Raw NWP", "DRIZZL"],
            y=[nwp, corrected],
            text=[
                f"{nwp:.1f} mm",
                f"{corrected:.1f} mm"
            ],
            textposition="outside",
            marker=dict(
                color=["#ff6b6b", "#00d4aa"],
                line=dict(
                    color="white",
                    width=2
                )
            )
        )
    )

    fig.update_layout(
        title={
            "text": f"📊 DRIZZL Correction: {correction:+.1f} mm",
            "x": 0.5
        },
        template="plotly_dark",
        height=500,
        showlegend=False,
        xaxis_title="",
        yaxis_title="Rainfall (mm)",
        font=dict(size=16),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

def probability_meter(probability):

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability,
            title={"text": "⚠️ Heavy Rain Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, 30], "color": "lightgreen"},
                    {"range": [30, 60], "color": "yellow"},
                    {"range": [60, 80], "color": "orange"},
                    {"range": [80, 100], "color": "red"}
                ]
            }
        )
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, width="stretch")

def regime_probability_chart(probabilities,labels):
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=[p * 100 for p in probabilities],
            hole=0.5,
            textinfo="label+percent",
            pull=[0.05, 0, 0]
        )
    )

    fig.update_layout(title="🌦 Regime Probabilities",template="plotly_dark",height=450,showlegend=True)
    st.plotly_chart(fig,width="stretch")