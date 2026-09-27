"""
Historical Replay / Performance Validation
Time-series cross-validation, rolling window evaluation, performance tracking
"""
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from verification import compute_all_metrics, print_metrics_report


def load_models():
    """Load trained models"""
    return {
        "classifier": joblib.load("models/regime_classifier.pkl"),
        "active": joblib.load("models/active_model.pkl"),
        "break": joblib.load("models/break_model.pkl"),
        "depression": joblib.load("models/depression_model.pkl"),
    }


def predict_rainfall(models, X_regime, X_full):
    """Make predictions using regime-specific models"""
    regime = models["classifier"].predict(X_regime)[0]
    probs = models["classifier"].predict_proba(X_regime)[0]
    confidence = max(probs) * 100
    
    if regime == "Active":
        corrected = models["active"].predict(X_full)[0]
    elif regime == "Break":
        corrected = models["break"].predict(X_full)[0]
    else:
        corrected = models["depression"].predict(X_full)[0]
    
    return corrected, regime, confidence


def rolling_window_validation(df, window_days=30, step_days=7, min_train_days=60):
    """
    Rolling window time-series cross-validation.
    Trains on past window, validates on next step.
    """
    df = df.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(df["date"]).unique()
    dates.sort()
    
    results = []
    
    for i in range(0, len(dates) - window_days - step_days, step_days):
        train_end = dates[i + window_days - 1]
        val_start = dates[i + window_days]
        val_end = dates[min(i + window_days + step_days - 1, len(dates) - 1)]
        
        train_mask = (pd.to_datetime(df["date"]) <= train_end) & \
                     (pd.to_datetime(df["date"]) >= dates[max(0, i + window_days - min_train_days)])
        val_mask = (pd.to_datetime(df["date"]) >= val_start) & \
                   (pd.to_datetime(df["date"]) <= val_end)
        
        train_df = df[train_mask]
        val_df = df[val_mask]
        
        if len(train_df) < 50 or len(val_df) < 10:
            continue
        
        # Train models on this window
        from train_models import train_regime_models
        models = train_regime_models(train_df)
        
        # Validate
        val_results = validate_on_data(models, val_df)
        val_results["train_period"] = f"{train_mask.idxmax()} to {train_end}"
        val_results["val_period"] = f"{val_start} to {val_end}"
        val_results["train_samples"] = len(train_df)
        val_results["val_samples"] = len(val_df)
        
        results.append(val_results)
        print(f"Window {len(results)}: Train={len(train_df)}, Val={len(val_df)}, "
              f"RMSE={val_results['continuous']['RMSE']:.2f}")
    
    return results


def train_regime_models(df):
    """Train all models on given dataframe"""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    
    X = df[["temperature", "humidity", "wind", "pressure"]]
    y = df["regime"]
    
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X, y)
    
    models = {"classifier": classifier}
    
    for regime_name in ["Active", "Break", "Depression"]:
        regime_df = df[df["regime"] == regime_name]
        if len(regime_df) < 10:
            continue
        
        X_reg = regime_df[["temperature", "humidity", "wind", "pressure", "nwp_rainfall"]]
        y_reg = regime_df["actual_rainfall"]
        
        reg_model = RandomForestRegressor(random_state=42)
        reg_model.fit(X_reg, y_reg)
        models[regime_name.lower()] = reg_model
    
    return models


def validate_on_data(models, df):
    """Run validation on a dataset"""
    predictions = []
    actuals = []
    regimes = []
    confidences = []
    
    for _, row in df.iterrows():
        X_regime = pd.DataFrame([[
            row["temperature"], row["humidity"], row["wind"], row["pressure"]
        ]], columns=["temperature", "humidity", "wind", "pressure"])
        
        X_full = pd.DataFrame([[
            row["temperature"], row["humidity"], row["wind"], 
            row["pressure"], row["nwp_rainfall"]
        ]], columns=["temperature", "humidity", "wind", "pressure", "nwp_rainfall"])
        
        corrected, regime, confidence = predict_rainfall(models, X_regime, X_full)
        
        predictions.append(corrected)
        actuals.append(row["actual_rainfall"])
        regimes.append(regime)
        confidences.append(confidence)
    
    y_true = np.array(actuals)
    y_pred = np.array(predictions)
    
    metrics = compute_all_metrics(y_true, y_pred)
    metrics["predictions"] = predictions
    metrics["actuals"] = actuals
    metrics["regimes"] = regimes
    
    return metrics


def historical_replay(df, models=None):
    """
    Full historical replay: run model on each day sequentially,
    track performance over time.
    """
    if models is None:
        models = load_models()
    
    df = df.sort_values("date").reset_index(drop=True)
    
    daily_results = []
    
    for date in pd.to_datetime(df["date"]).unique():
        day_df = df[pd.to_datetime(df["date"]) == date]
        
        predictions = []
        actuals = []
        
        for _, row in day_df.iterrows():
            X_regime = pd.DataFrame([[
                row["temperature"], row["humidity"], row["wind"], row["pressure"]
            ]], columns=["temperature", "humidity", "wind", "pressure"])
            
            X_full = pd.DataFrame([[
                row["temperature"], row["humidity"], row["wind"], 
                row["pressure"], row["nwp_rainfall"]
            ]], columns=["temperature", "humidity", "wind", "pressure", "nwp_rainfall"])
            
            corrected, regime, confidence = predict_rainfall(models, X_regime, X_full)
            predictions.append(corrected)
            actuals.append(row["actual_rainfall"])
        
        if len(predictions) > 0:
            y_true = np.array(actuals)
            y_pred = np.array(predictions)
            
            metrics = compute_all_metrics(y_true, y_pred)
            metrics["date"] = str(date)
            metrics["n_districts"] = len(predictions)
            metrics["mean_predicted"] = np.mean(predictions)
            metrics["mean_actual"] = np.mean(actuals)
            
            daily_results.append(metrics)
    
    return daily_results


def performance_summary(daily_results):
    """Aggregate daily results into performance summary"""
    if not daily_results:
        return {}
    
    summary = {
        "period": f"{daily_results[0]['date']} to {daily_results[-1]['date']}",
        "n_days": len(daily_results),
        "continuous": {},
        "categorical": {}
    }
    
    # Average continuous metrics
    for metric in ["RMSE", "MAE", "Bias", "Correlation"]:
        vals = [d["continuous"][metric] for d in daily_results if not np.isnan(d["continuous"][metric])]
        if vals:
            summary["continuous"][metric] = {
                "mean": np.mean(vals),
                "std": np.std(vals),
                "min": np.min(vals),
                "max": np.max(vals)
            }
    
    # Average categorical metrics per threshold
    thresholds = daily_results[0]["categorical"].keys()
    for thresh in thresholds:
        summary["categorical"][thresh] = {}
        for metric in ["POD", "FAR", "CSI", "ETS", "HSS", "FSS"]:
            vals = [d["categorical"][thresh][metric] for d in daily_results 
                    if not np.isnan(d["categorical"][thresh][metric])]
            if vals:
                summary["categorical"][thresh][metric] = {
                    "mean": np.mean(vals),
                    "std": np.std(vals)
                }
    
    return summary


def save_performance_log(results, path="logs/performance_log.json"):
    """Save performance tracking log"""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Convert numpy types to native Python for JSON
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
        json.dump(convert(results), f, indent=2)
    print(f"Performance log saved to {path}")


def load_performance_log(path="logs/performance_log.json"):
    """Load performance tracking log"""
    with open(path, "r") as f:
        return json.load(f)


def plot_performance_trends(daily_results, save_path=None):
    """Plot performance metrics over time"""
    import matplotlib.pyplot as plt
    
    dates = [d["date"] for d in daily_results]
    rmse_vals = [d["continuous"]["RMSE"] for d in daily_results]
    csi_vals = [d["categorical"]["threshold_15mm"]["CSI"] for d in daily_results]
    pod_vals = [d["categorical"]["threshold_15mm"]["POD"] for d in daily_results]
    far_vals = [d["categorical"]["threshold_15mm"]["FAR"] for d in daily_results]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(dates, rmse_vals, 'b-o', markersize=3)
    axes[0, 0].set_title("RMSE Over Time")
    axes[0, 0].set_ylabel("RMSE (mm)")
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(dates, csi_vals, 'g-o', markersize=3, label="CSI")
    axes[0, 1].plot(dates, pod_vals, 'r-o', markersize=3, label="POD")
    axes[0, 1].plot(dates, far_vals, 'm-o', markersize=3, label="FAR")
    axes[0, 1].set_title("Categorical Metrics (15mm threshold)")
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Mean predicted vs actual
    mean_pred = [d["mean_predicted"] for d in daily_results]
    mean_act = [d["mean_actual"] for d in daily_results]
    axes[1, 0].plot(dates, mean_pred, 'b-o', markersize=3, label="Predicted")
    axes[1, 0].plot(dates, mean_act, 'r-o', markersize=3, label="Actual")
    axes[1, 0].set_title("Mean Daily Rainfall")
    axes[1, 0].set_ylabel("Rainfall (mm)")
    axes[1, 0].legend()
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Bias over time
    bias_vals = [d["continuous"]["Bias"] for d in daily_results]
    axes[1, 1].plot(dates, bias_vals, 'k-o', markersize=3)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_title("Bias Over Time (Positive = Overprediction)")
    axes[1, 1].set_ylabel("Bias (mm)")
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # Demo: load data and run replay
    df = pd.read_csv("data/weather_data.csv")
    df["date"] = pd.date_range("2023-06-01", periods=len(df), freq="D").astype(str)
    
    models = load_models()
    results = historical_replay(df, models)
    
    summary = performance_summary(results)
    print(json.dumps(summary, indent=2, default=str))
    
    save_performance_log({"daily": results, "summary": summary})