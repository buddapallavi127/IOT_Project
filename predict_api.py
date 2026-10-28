import pandas as pd
import joblib

# Load saved model + feature list once
model = joblib.load("data/ml/congestion_classifier.pkl")
feature_cols = joblib.load("data/ml/feature_cols.pkl")

def predict_congestion(current_data: dict) -> dict:
    """
    current_data must have keys matching feature_cols, e.g.:
    {
        "speed_lag1": 10.2, "queue_lag1": 3, "occ_lag1": 0.4,
        "speed_lag2": 10.5, "queue_lag2": 2, "occ_lag2": 0.35,
        "speed_lag3": 11.0, "queue_lag3": 2, "occ_lag3": 0.30,
        "hour": 8, "free_flow_speed": 13.9
    }
    Returns predicted congestion 5 min ahead: 0 = clear, 1 = congested
    """
    X = pd.DataFrame([current_data])[feature_cols]
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]  # probability of congestion
    return {"congested_5min": int(pred), "confidence": float(prob)}

# quick test
if __name__ == "__main__":
    example = {
        "speed_lag1": 5.0, "queue_lag1": 8, "occ_lag1": 0.6,
        "speed_lag2": 5.5, "queue_lag2": 7, "occ_lag2": 0.55,
        "speed_lag3": 6.0, "queue_lag3": 6, "occ_lag3": 0.50,
        "hour": 8, "free_flow_speed": 13.9
    }
    print(predict_congestion(example))