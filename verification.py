"""
Verification Metrics for Rainfall Forecasts
RMSE, ETS, CSI, POD, FAR, FSS (Fractional Skill Score)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    """Root Mean Square Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))


def bias(y_true, y_pred):
    """Mean Bias Error (positive = overprediction)"""
    return np.mean(y_pred - y_true)


def correlation(y_true, y_pred):
    """Pearson correlation coefficient"""
    return np.corrcoef(y_true, y_pred)[0, 1]


# Categorical verification metrics (for rainfall thresholds)
def contingency_table(y_true, y_pred, threshold):
    """
    Build 2x2 contingency table for binary event (rainfall >= threshold)
    Returns: hits, misses, false_alarms, correct_negatives
    """
    obs_yes = y_true >= threshold
    fcst_yes = y_pred >= threshold
    
    hits = np.sum(obs_yes & fcst_yes)
    misses = np.sum(obs_yes & ~fcst_yes)
    false_alarms = np.sum(~obs_yes & fcst_yes)
    correct_negatives = np.sum(~obs_yes & ~fcst_yes)
    
    return hits, misses, false_alarms, correct_negatives


def pod(y_true, y_pred, threshold):
    """Probability of Detection (Hit Rate)"""
    hits, misses, _, _ = contingency_table(y_true, y_pred, threshold)
    if hits + misses == 0:
        return np.nan
    return hits / (hits + misses)


def far(y_true, y_pred, threshold):
    """False Alarm Ratio"""
    hits, _, false_alarms, _ = contingency_table(y_true, y_pred, threshold)
    if hits + false_alarms == 0:
        return np.nan
    return false_alarms / (hits + false_alarms)


def csi(y_true, y_pred, threshold):
    """Critical Success Index (Threat Score)"""
    hits, misses, false_alarms, _ = contingency_table(y_true, y_pred, threshold)
    if hits + misses + false_alarms == 0:
        return np.nan
    return hits / (hits + misses + false_alarms)


def ets(y_true, y_pred, threshold):
    """Equitable Threat Score (Gilbert Skill Score)"""
    hits, misses, false_alarms, correct_negatives = contingency_table(y_true, y_pred, threshold)
    n = hits + misses + false_alarms + correct_negatives
    if n == 0:
        return np.nan
    
    hits_random = (hits + misses) * (hits + false_alarms) / n
    if hits + misses + false_alarms - hits_random == 0:
        return np.nan
    return (hits - hits_random) / (hits + misses + false_alarms - hits_random)


def hss(y_true, y_pred, threshold):
    """Heidke Skill Score"""
    hits, misses, false_alarms, correct_negatives = contingency_table(y_true, y_pred, threshold)
    n = hits + misses + false_alarms + correct_negatives
    if n == 0:
        return np.nan
    
    expected_correct = ((hits + misses) * (hits + false_alarms) + 
                        (misses + correct_negatives) * (false_alarms + correct_negatives)) / n
    if n - expected_correct == 0:
        return np.nan
    return (hits + correct_negatives - expected_correct) / (n - expected_correct)


def fss(y_true, y_pred, threshold, window_size=3):
    """
    Fractional Skill Score (spatial verification)
    For single point, reduces to standard skill score.
    For gridded data, applies neighborhood averaging.
    """
    # For single time series, use a temporal window
    # Convert to binary fields
    obs_bin = (y_true >= threshold).astype(float)
    fcst_bin = (y_pred >= threshold).astype(float)
    
    if len(obs_bin) < window_size:
        return np.nan
    
    # Apply moving average (neighborhood)
    from scipy.ndimage import uniform_filter1d
    obs_smooth = uniform_filter1d(obs_bin, size=window_size, mode='constant')
    fcst_smooth = uniform_filter1d(fcst_bin, size=window_size, mode='constant')
    
    mse = np.mean((obs_smooth - fcst_smooth) ** 2)
    mse_ref = np.mean(obs_smooth ** 2) + np.mean(fcst_smooth ** 2)
    
    if mse_ref == 0:
        return 1.0
    return 1 - mse / mse_ref


def compute_all_metrics(y_true, y_pred, thresholds=[2.5, 15, 65, 115]):
    """
    Compute all verification metrics for multiple thresholds.
    Returns dict with continuous and categorical metrics.
    """
    metrics = {
        "continuous": {
            "RMSE": rmse(y_true, y_pred),
            "MAE": mae(y_true, y_pred),
            "Bias": bias(y_true, y_pred),
            "Correlation": correlation(y_true, y_pred),
        },
        "categorical": {}
    }
    
    for thresh in thresholds:
        metrics["categorical"][f"threshold_{thresh}mm"] = {
            "POD": pod(y_true, y_pred, thresh),
            "FAR": far(y_true, y_pred, thresh),
            "CSI": csi(y_true, y_pred, thresh),
            "ETS": ets(y_true, y_pred, thresh),
            "HSS": hss(y_true, y_pred, thresh),
            "FSS": fss(y_true, y_pred, thresh),
        }
    
    return metrics


def print_metrics_report(metrics):
    """Pretty print verification metrics"""
    print("=" * 60)
    print("VERIFICATION METRICS REPORT")
    print("=" * 60)
    
    print("\n📊 CONTINUOUS METRICS:")
    for name, val in metrics["continuous"].items():
        print(f"  {name:12s}: {val:.4f}")
    
    print("\n📈 CATEGORICAL METRICS (per threshold):")
    for thresh_name, vals in metrics["categorical"].items():
        print(f"\n  {thresh_name}:")
        for name, val in vals.items():
            if not np.isnan(val):
                print(f"    {name:4s}: {val:.4f}")
            else:
                print(f"    {name:4s}: N/A")


def bootstrap_ci(y_true, y_pred, metric_func, n_bootstrap=1000, alpha=0.05):
    """Compute bootstrap confidence interval for a metric"""
    n = len(y_true)
    values = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        values.append(metric_func(y_true[idx], y_pred[idx]))
    
    lower = np.percentile(values, 100 * alpha / 2)
    upper = np.percentile(values, 100 * (1 - alpha / 2))
    return lower, upper


if __name__ == "__main__":
    # Demo with synthetic data
    np.random.seed(42)
    y_true = np.random.exponential(20, 100)
    y_pred = y_true + np.random.normal(0, 10, 100)
    y_pred = np.maximum(y_pred, 0)
    
    metrics = compute_all_metrics(y_true, y_pred)
    print_metrics_report(metrics)