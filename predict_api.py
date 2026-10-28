#!/usr/bin/env python3
"""
predict_api.py  --  Inference API for Real-Time Congestion Prediction
=====================================================================
Uses the trained XGBoost model to predict whether an edge approach
will become congested 5 minutes in the future.
"""

import json
import os
import joblib
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(ROOT, "data", "ml", "congestion_classifier.pkl")
FEAT_FILE  = os.path.join(ROOT, "data", "ml", "feature_cols.pkl")
META_FILE  = os.path.join(ROOT, "data", "ml", "model_meta.json")

# Load model, features, and metadata
model = joblib.load(MODEL_FILE)
feature_cols = joblib.load(FEAT_FILE)
threshold = 0.50
if os.path.exists(META_FILE):
    try:
        with open(META_FILE) as f:
            meta = json.load(f)
            threshold = meta.get("decision_threshold", 0.50)
    except Exception:
        pass


def predict_congestion(data: dict) -> dict:
    """
    Given current traffic metrics on an edge, predicts if it will be
    congested 5 minutes from now.
    
    Can take raw metric lags or complete feature vector.
    """
    row = dict(data)

    # Compute rolling and trend features if not directly provided
    if "speed_roll3" not in row and all(k in row for k in ["speed_lag1", "speed_lag2", "speed_lag3"]):
        row["speed_roll3"] = (row["speed_lag1"] + row["speed_lag2"] + row["speed_lag3"]) / 3.0
    if "queue_roll3" not in row and all(k in row for k in ["queue_lag1", "queue_lag2", "queue_lag3"]):
        row["queue_roll3"] = (row["queue_lag1"] + row["queue_lag2"] + row["queue_lag3"]) / 3.0
    if "speed_trend" not in row and "speed_lag1" in row and "speed_lag2" in row:
        row["speed_trend"] = row["speed_lag1"] - row["speed_lag2"]
    if "queue_trend" not in row and "queue_lag1" in row and "queue_lag2" in row:
        row["queue_trend"] = row["queue_lag1"] - row["queue_lag2"]
    if "speed_ratio" not in row and "speed_lag1" in row and "free_flow_speed" in row:
        row["speed_ratio"] = row["speed_lag1"] / (row["free_flow_speed"] + 1e-5)

    X = pd.DataFrame([row])[feature_cols]
    prob = float(model.predict_proba(X)[0][1])
    is_congested = int(prob >= threshold)

    return {
        "congested_5min": is_congested,
        "probability": round(prob, 3),
        "decision_threshold": round(threshold, 2),
        "status": "CONGESTED" if is_congested else "CLEAR"
    }


if __name__ == "__main__":
    print("=" * 65)
    print(f"  🚦 Testing Congestion Prediction API (Decision Thresh: {threshold:.2f})")
    print("=" * 65)

    # Case 1: Free-flow, uncongested approach (high speed, 0 queue)
    uncongested_input = {
        "speed_lag1": 13.5, "speed_lag2": 13.8, "speed_lag3": 13.9,
        "queue_lag1": 0,    "queue_lag2": 0,    "queue_lag3": 0,
        "occ_lag1": 0.05,   "occ_lag2": 0.04,   "occ_lag3": 0.04,
        "hour": 10, "free_flow_speed": 13.89
    }
    res1 = predict_congestion(uncongested_input)
    print("\nCase 1 [Free Flow / Normal Traffic]:")
    print(f"  Input  : speed=13.5 m/s, queue=0, occ=5%")
    print(f"  Result : Status={res1['status']} (congested_5min={res1['congested_5min']}, prob={res1['probability']})")
    assert res1["congested_5min"] == 0, "Expected clear traffic"

    # Case 2: Heavy bottleneck, rapidly forming queue (low speed, high queue)
    congested_input = {
        "speed_lag1": 2.1,  "speed_lag2": 3.8,  "speed_lag3": 7.0,
        "queue_lag1": 12,   "queue_lag2": 9,    "queue_lag3": 5,
        "occ_lag1": 0.72,   "occ_lag2": 0.58,   "occ_lag3": 0.35,
        "hour": 17, "free_flow_speed": 13.89
    }
    res2 = predict_congestion(congested_input)
    print("\nCase 2 [Rush-Hour Jam / Queue Spilling Back]:")
    print(f"  Input  : speed=2.1 m/s (dropping from 7.0), queue=12 veh (surging), occ=72%")
    print(f"  Result : Status={res2['status']} (congested_5min={res2['congested_5min']}, prob={res2['probability']})")
    assert res2["congested_5min"] == 1, "Expected congested traffic"

    print("\n✅ Verification successful: API correctly discriminates clear vs congested states.")