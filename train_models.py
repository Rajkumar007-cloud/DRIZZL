import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("data/weather_data.csv")

# ---------------------
# REGIME CLASSIFIER
# ---------------------

X = df[
    [
        "temperature",
        "humidity",
        "wind",
        "pressure"
    ]
]

y = df["regime"]

classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

classifier.fit(X, y)

joblib.dump(
    classifier,
    "models/regime_classifier.pkl"
)

# ---------------------
# ACTIVE MODEL
# ---------------------

active_df = df[df["regime"] == "Active"]

X_active = active_df[
    [
        "temperature",
        "humidity",
        "wind",
        "pressure",
        "nwp_rainfall"
    ]
]

y_active = active_df["actual_rainfall"]

active_model = RandomForestRegressor()

active_model.fit(
    X_active,
    y_active
)

joblib.dump(
    active_model,
    "models/active_model.pkl"
)

# ---------------------
# BREAK MODEL
# ---------------------

break_df = df[df["regime"] == "Break"]

X_break = break_df[
    [
        "temperature",
        "humidity",
        "wind",
        "pressure",
        "nwp_rainfall"
    ]
]

y_break = break_df["actual_rainfall"]

break_model = RandomForestRegressor()

break_model.fit(
    X_break,
    y_break
)

joblib.dump(
    break_model,
    "models/break_model.pkl"
)

# ---------------------
# DEPRESSION MODEL
# ---------------------

dep_df = df[df["regime"] == "Depression"]

X_dep = dep_df[
    [
        "temperature",
        "humidity",
        "wind",
        "pressure",
        "nwp_rainfall"
    ]
]

y_dep = dep_df["actual_rainfall"]

dep_model = RandomForestRegressor()

dep_model.fit(
    X_dep,
    y_dep
)

joblib.dump(
    dep_model,
    "models/depression_model.pkl"
)

print("Models trained!")


def train_regime_models(df):
    """Train all models on given dataframe - for historical replay validation"""
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