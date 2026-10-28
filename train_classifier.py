# ============================================
# STEP 1: Load libraries
# ============================================
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# ============================================
# STEP 2: Load and prepare data
# ============================================
df = pd.read_csv("data/ml/traffic_long.csv")
df = df.sort_values(["run", "edge_id", "time"])
group = df.groupby(["run", "edge_id"])

for lag in [1, 2, 3]:
    df[f"speed_lag{lag}"] = group["speed_ms"].shift(lag)
    df[f"queue_lag{lag}"] = group["queue_vehicles"].shift(lag)
    df[f"occ_lag{lag}"] = group["occupancy"].shift(lag)

df["hour"] = (df["time"] // 3600) % 24

# Target: will it be congested 5 min from now?
df["target_congested_5min"] = group["congested"].shift(-5)

df = df.dropna()

feature_cols = [c for c in df.columns if "lag" in c] + ["hour", "free_flow_speed"]

# ============================================
# STEP 3: Split by seed
# ============================================
train = df[df.seed.isin([1, 2])]
test = df[df.seed == 3]

X_train, y_train = train[feature_cols], train["target_congested_5min"]
X_test, y_test = test[feature_cols], test["target_congested_5min"]

# ============================================
# STEP 4: Compute class imbalance (needed for XGBoost)
# ============================================
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
print("Not congested:", neg, " Congested:", pos, " Ratio:", neg/pos)

# ============================================
# STEP 5: Train both classifiers (now balanced)
# ============================================
rf = RandomForestClassifier(
    n_estimators=200, max_depth=12, n_jobs=-1, random_state=42,
    class_weight="balanced"
)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

xgb = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05, n_jobs=-1, random_state=42,
    scale_pos_weight=neg/pos
)
xgb.fit(X_train, y_train)
pred_xgb = xgb.predict(X_test)

# ============================================
# STEP 6: Evaluate
# ============================================
for name, pred in [("Random Forest", pred_rf), ("XGBoost", pred_xgb)]:
    print(f"\n--- {name} (congestion classifier, balanced) ---")
    print("Accuracy :", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred))
    print("Recall   :", recall_score(y_test, pred))
    print("F1       :", f1_score(y_test, pred))

# ============================================
# STEP 7: Save the best model
# ============================================
joblib.dump(xgb, "data/ml/congestion_classifier.pkl")
joblib.dump(feature_cols, "data/ml/feature_cols.pkl")
print("\nSaved model to data/ml/congestion_classifier.pkl")