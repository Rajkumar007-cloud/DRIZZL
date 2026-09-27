"""
District-Level Rainfall Forecast Product
Generates forecasts for multiple districts with spatial visualization
"""
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta
from real_data import DISTRICT_COORDS, fetch_era5_reanalysis, fetch_nwp_forecast


def load_models():
    """Load trained models"""
    return {
        "classifier": joblib.load("models/regime_classifier.pkl"),
        "active": joblib.load("models/active_model.pkl"),
        "break": joblib.load("models/break_model.pkl"),
        "depression": joblib.load("models/depression_model.pkl"),
    }


def predict_district(models, district, weather_data=None):
    """
    Generate forecast for a single district.
    If weather_data not provided, fetches synthetic data.
    """
    coords = DISTRICT_COORDS.get(district)
    if not coords:
        raise ValueError(f"Unknown district: {district}")
    
    if weather_data is None:
        # Fetch current conditions
        era5 = fetch_era5_reanalysis(district, datetime.now())
        nwp = fetch_nwp_forecast(district, datetime.now())
        weather_data = {
            "temperature": era5["temperature"],
            "humidity": era5["humidity"],
            "wind": era5["wind_speed"],
            "pressure": era5["pressure"],
            "nwp_rainfall": nwp["nwp_rainfall"],
        }
    
    # Regime classification
    X_regime = pd.DataFrame([[
        weather_data["temperature"], 
        weather_data["humidity"], 
        weather_data["wind"], 
        weather_data["pressure"]
    ]], columns=["temperature", "humidity", "wind", "pressure"])
    
    regime = models["classifier"].predict(X_regime)[0]
    probabilities = models["classifier"].predict_proba(X_regime)[0]
    confidence = max(probabilities) * 100
    reliability = min(100, confidence * 0.9)
    
    # Rainfall correction
    X_full = pd.DataFrame([[
        weather_data["temperature"],
        weather_data["humidity"],
        weather_data["wind"],
        weather_data["pressure"],
        weather_data["nwp_rainfall"]
    ]], columns=["temperature", "humidity", "wind", "pressure", "nwp_rainfall"])
    
    if regime == "Active":
        corrected = models["active"].predict(X_full)[0]
    elif regime == "Break":
        corrected = models["break"].predict(X_full)[0]
    else:
        corrected = models["depression"].predict(X_full)[0]
    
    # IMD category
    if corrected < 15:
        category = "Light Rain"
        category_emoji = "🌦"
    elif corrected < 65:
        category = "Moderate Rain"
        category_emoji = "🌧"
    elif corrected < 115:
        category = "Heavy Rain"
        category_emoji = "🌧"
    elif corrected < 205:
        category = "Very Heavy Rain"
        category_emoji = "⛈"
    else:
        category = "Extreme Rainfall"
        category_emoji = "🚨"
    
    # Risk
    from risk import calculate_risk_score, get_risk_level, get_risk_message
    risk_score = calculate_risk_score(corrected, confidence, reliability)
    risk_level = get_risk_level(risk_score)
    risk_message = get_risk_message(risk_score)
    
    return {
        "district": district,
        "state": coords["state"],
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "timestamp": datetime.now().isoformat(),
        "weather": weather_data,
        "regime": regime,
        "regime_probabilities": dict(zip(models["classifier"].classes_, probabilities)),
        "regime_confidence": confidence,
        "forecast_reliability": reliability,
        "nwp_rainfall": weather_data["nwp_rainfall"],
        "corrected_rainfall": corrected,
        "imd_category": category,
        "imd_category_emoji": category_emoji,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_message": risk_message,
    }


def forecast_all_districts(models=None, weather_data_dict=None):
    """
    Generate forecasts for all districts.
    weather_data_dict: optional dict of district -> weather data
    """
    if models is None:
        models = load_models()
    
    results = []
    for district in DISTRICT_COORDS.keys():
        weather_data = None
        if weather_data_dict and district in weather_data_dict:
            weather_data = weather_data_dict[district]
        
        try:
            forecast = predict_district(models, district, weather_data)
            results.append(forecast)
        except Exception as e:
            print(f"Error forecasting {district}: {e}")
            results.append({
                "district": district,
                "error": str(e)
            })
    
    return results


def create_district_geojson(forecasts):
    """Create GeoJSON for map visualization"""
    features = []
    for fcst in forecasts:
        if "error" in fcst:
            continue
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [fcst["longitude"], fcst["latitude"]]
            },
            "properties": {
                "district": fcst["district"],
                "state": fcst["state"],
                "corrected_rainfall": round(fcst["corrected_rainfall"], 1),
                "nwp_rainfall": round(fcst["nwp_rainfall"], 1),
                "regime": fcst["regime"],
                "regime_confidence": round(fcst["regime_confidence"], 1),
                "imd_category": fcst["imd_category"],
                "imd_category_emoji": fcst["imd_category_emoji"],
                "risk_level": fcst["risk_level"],
                "risk_score": fcst["risk_score"],
            }
        })
    
    return {
        "type": "FeatureCollection",
        "features": features
    }


def save_district_forecast(forecasts, path="outputs/district_forecast.json"):
    """Save district forecasts to JSON"""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    output = {
        "timestamp": datetime.now().isoformat(),
        "forecasts": forecasts,
        "geojson": create_district_geojson(forecasts)
    }
    
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    with open(path, "w") as f:
        json.dump(convert(output), f, indent=2)
    print(f"District forecast saved to {path}")


def load_district_forecast(path="outputs/district_forecast.json"):
    """Load district forecasts from JSON"""
    with open(path, "r") as f:
        return json.load(f)


def print_district_summary(forecasts):
    """Print formatted district forecast summary"""
    print(f"\n{'='*80}")
    print(f"DISTRICT-LEVEL RAINFALL FORECAST - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*80}")
    
    # Sort by corrected rainfall descending
    valid = [f for f in forecasts if "error" not in f]
    valid.sort(key=lambda x: x["corrected_rainfall"], reverse=True)
    
    print(f"{'District':<20} {'State':<15} {'NWP':>6} {'DRIZZL':>7} {'Regime':<10} {'Conf%':>6} {'Category':<20} {'Risk':<12}")
    print("-" * 110)
    
    for fcst in valid:
        print(f"{fcst['district']:<20} {fcst['state']:<15} "
              f"{fcst['nwp_rainfall']:>6.1f} {fcst['corrected_rainfall']:>7.1f} "
              f"{fcst['regime']:<10} {fcst['regime_confidence']:>6.1f} "
              f"{fcst['imd_category_emoji']} {fcst['imd_category']:<18} "
              f"{fcst['risk_level']:<12}")
    
    # Summary stats
    rainfalls = [f["corrected_rainfall"] for f in valid]
    print(f"\nSummary: {len(valid)} districts | "
          f"Mean: {np.mean(rainfalls):.1f}mm | "
          f"Max: {np.max(rainfalls):.1f}mm ({valid[0]['district']}) | "
          f"Min: {np.min(rainfalls):.1f}mm")
    
    # Regime distribution
    regimes = [f["regime"] for f in valid]
    for r in ["Active", "Break", "Depression"]:
        count = regimes.count(r)
        print(f"  {r}: {count} districts")


if __name__ == "__main__":
    models = load_models()
    forecasts = forecast_all_districts(models)
    print_district_summary(forecasts)
    save_district_forecast(forecasts)