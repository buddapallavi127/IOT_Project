#!/usr/bin/env python3
"""
train_classifier.py  --  Train ML Congestion Classifiers (Random Forest & XGBoost)
=================================================================================
Predicts traffic congestion 5 minutes ahead on approach edges.

Features:
  - Multi-step lag features (speed, queue, occupancy)
  - Rolling 3-step averages (speed_roll3, queue_roll3)
  - Momentum / trend features (speed_trend, queue_trend)
  - Capacity ratio (speed / free_flow_speed)
  - Temporal feature (hour of day)

Handles class imbalance (~9:1) via:
  - scale_pos_weight for XGBoost
  - balanced_subsample for Random Forest
  - F1-optimal threshold tuning on validation/train distribution

Evaluation metrics reported:
  - PR-AUC (Precision-Recall Area Under Curve, critical for imbalanced data)
  - ROC-AUC
  - Precision, Recall, F1 score
  - Confusion Matrix
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

ROOT = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.join(ROOT, "data", "ml")
DATA_CSV = os.path.join(ML_DIR, "traffic_long.csv")

def main():
    print("=" * 65)
    print("  🚦 Training ML Congestion Prediction Classifiers")
    print("=" * 65)

    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"Missing {DATA_CSV}. Run 'python traffic_sim.py all' first.")

    # 1. Load data
    df = pd.read_csv(DATA_CSV)
    df = df.sort_values(["run", "edge_id", "time"])
    group = df.groupby(["run", "edge_id"])

    # 2. Engineer features
    for lag in [1, 2, 3]:
        df[f"speed_lag{lag}"] = group["speed_ms"].shift(lag)
        df[f"queue_lag{lag}"] = group["queue_vehicles"].shift(lag)
        df[f"occ_lag{lag}"] = group["occupancy"].shift(lag)

    df["speed_roll3"] = (df["speed_lag1"] + df["speed_lag2"] + df["speed_lag3"]) / 3.0
    df["queue_roll3"] = (df["queue_lag1"] + df["queue_lag2"] + df["queue_lag3"]) / 3.0
    df["speed_trend"] = df["speed_lag1"] - df["speed_lag2"]
    df["queue_trend"] = df["queue_lag1"] - df["queue_lag2"]
    df["speed_ratio"] = df["speed_lag1"] / (df["free_flow_speed"] + 1e-5)
    df["hour"] = (df["time"] // 3600) % 24

    # Target: congestion 5 minutes ahead (1 step forward in 300s bin dataset)
    df["target_congested_5min"] = group["congested"].shift(-1)
    df = df.dropna()

    feature_cols = [
        "speed_lag1", "speed_lag2", "speed_lag3",
        "queue_lag1", "queue_lag2", "queue_lag3",
        "occ_lag1", "occ_lag2", "occ_lag3",
        "speed_roll3", "queue_roll3",
        "speed_trend", "queue_trend", "speed_ratio",
        "hour", "free_flow_speed"
    ]

    # 3. Train/Test split by random seed (unseen demand distribution)
    train = df[df.seed.isin([1, 2])]
    test = df[df.seed == 3]

    X_train = train[feature_cols]
    y_train = train["target_congested_5min"].astype(int)
    X_test  = test[feature_cols]
    y_test  = test["target_congested_5min"].astype(int)

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    imbalance_ratio = neg / pos

    print(f"Dataset summary:")
    print(f"  Training set (Seeds 1, 2): {len(X_train)} samples [Clear: {neg}, Congested: {pos}, Ratio: {imbalance_ratio:.2f}:1]")
    print(f"  Test set     (Seed 3)    : {len(X_test)} samples [Clear: {(y_test==0).sum()}, Congested: {(y_test==1).sum()}]")

    # 4. Train Random Forest (balanced subsample)
    print("\n[1/2] Training Balanced Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    rf_pred = rf.predict(X_test)

    # 5. Train XGBoost with scale_pos_weight
    print("[2/2] Training XGBoost Classifier...")
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=imbalance_ratio,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    xgb_probs = xgb.predict_proba(X_test)[:, 1]

    # Optimal threshold search for F1 maximization on training/validation distribution
    train_probs = xgb.predict_proba(X_train)[:, 1]
    best_th, best_f1 = 0.5, 0.0
    for th in np.linspace(0.3, 0.8, 51):
        f = f1_score(y_train, (train_probs >= th).astype(int))
        if f > best_f1:
            best_f1, best_th = f, th

    xgb_pred_tuned = (xgb_probs >= best_th).astype(int)

    # 6. Comprehensive Metrics Report
    print("\n" + "=" * 65)
    print("  📊 MODEL EVALUATION RESULTS (Unseen Seed 3)")
    print("=" * 65)

    def print_metrics(name, y_true, y_pred, y_probs, th_note=""):
        pr_auc = average_precision_score(y_true, y_probs)
        roc_auc = roc_auc_score(y_true, y_probs)
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        print(f"\n--- {name} {th_note}---")
        print(f"  PR-AUC (Precision-Recall Area) : {pr_auc:.3f}   <-- Key imbalanced metric")
        print(f"  ROC-AUC                        : {roc_auc:.3f}")
        print(f"  F1 Score                       : {f1:.3f}")
        print(f"  Recall (Congestion detected)   : {rec:.3f} ({cm[1,1]}/{cm[1,1]+cm[1,0]})")
        print(f"  Precision (True alarms)        : {prec:.3f}")
        print(f"  Balanced Accuracy              : {bal_acc:.3f}")
        print(f"  Overall Accuracy               : {acc:.3f}")
        print(f"  Confusion Matrix (TN, FP / FN, TP):")
        print(f"    [[{cm[0,0]:>4}, {cm[0,1]:>4}],")
        print(f"     [{cm[1,0]:>4}, {cm[1,1]:>4}]]")

    print_metrics("Random Forest", y_test, rf_pred, rf_probs, "(Threshold = 0.50) ")
    print_metrics("XGBoost", y_test, xgb_pred_tuned, xgb_probs, f"(Tuned Threshold = {best_th:.2f}) ")

    # 7. Save model and metadata
    model_path = os.path.join(ML_DIR, "congestion_classifier.pkl")
    feat_path = os.path.join(ML_DIR, "feature_cols.pkl")
    meta_path = os.path.join(ML_DIR, "model_meta.json")

    joblib.dump(xgb, model_path)
    joblib.dump(feature_cols, feat_path)
    with open(meta_path, "w") as f:
        json.dump({
            "model_type": "XGBClassifier",
            "decision_threshold": float(best_th),
            "features": feature_cols,
            "pr_auc": float(average_precision_score(y_test, xgb_probs)),
            "f1_score": float(f1_score(y_test, xgb_pred_tuned)),
            "recall": float(recall_score(y_test, xgb_pred_tuned)),
            "precision": float(precision_score(y_test, xgb_pred_tuned)),
        }, f, indent=2)

    print(f"\n✅ Saved trained XGBoost model to {model_path}")
    print(f"✅ Saved feature metadata to {meta_path}")

if __name__ == "__main__":
    main()