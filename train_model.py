# ============================================
# STEP 1: Load the libraries we need
# ============================================
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================
# STEP 2: Load the data
# ============================================
df = pd.read_csv("data/ml/traffic_long.csv")

print("Shape of data:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# Find out how far apart each time step is (bin size)
time_diffs = df.groupby(["run", "edge_id"])["time"].diff().dropna()
bin_seconds = time_diffs.mode()[0]
print("Bin size (seconds):", bin_seconds)

# ============================================
# STEP 3: Sort data so lag features make sense
# ============================================
df = df.sort_values(["run", "edge_id", "time"])
group = df.groupby(["run", "edge_id"])

# ============================================
# STEP 4: Create "lag" features (past values)
# These tell the model: "what was speed/queue 1,2,3 steps ago?"
# ============================================
for lag in [1, 2, 3]:
    df[f"speed_lag{lag}"] = group["speed_ms"].shift(lag)
    df[f"queue_lag{lag}"] = group["queue_vehicles"].shift(lag)
    df[f"occ_lag{lag}"] = group["occupancy"].shift(lag)

df["hour"] = (df["time"] // 3600) % 24

# ============================================
# STEP 5: Create the TARGETS (future speed we want to predict)
# Convert "5/10/15 minutes ahead" into number of steps using bin_seconds
# ============================================
steps_5min = max(1, round(300 / bin_seconds))
steps_10min = max(1, round(600 / bin_seconds))
steps_15min = max(1, round(900 / bin_seconds))

HORIZONS = {"5min": steps_5min, "10min": steps_10min, "15min": steps_15min}
print("Horizons (in steps):", HORIZONS)

for name, h in HORIZONS.items():
    df[f"target_{name}"] = group["speed_ms"].shift(-h)

# Drop rows with missing values (start/end of each run)
df = df.dropna()
print("Shape after cleanup:", df.shape)

# ============================================
# STEP 6: Pick which columns are inputs (features)
# ============================================
feature_cols = [c for c in df.columns if "lag" in c] + ["hour", "free_flow_speed"]
print("Features used:", feature_cols)

# ============================================
# STEP 7: Split into train/test by SEED (not random!)
# seeds 1,2 = train, seed 3 = test
# ============================================
train = df[df.seed.isin([1, 2])]
test = df[df.seed == 3]
print("Train rows:", len(train), "Test rows:", len(test))

# ============================================
# STEP 8: Train models + check accuracy for each horizon
# ============================================
results = {}

for name in HORIZONS:
    X_train, y_train = train[feature_cols], train[f"target_{name}"]
    X_test, y_test = test[feature_cols], test[f"target_{name}"]

    rf = RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)

    xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, n_jobs=-1, random_state=42)
    xgb.fit(X_train, y_train)
    pred_xgb = xgb.predict(X_test)

    results[name] = {
        "RF_MAE": mean_absolute_error(y_test, pred_rf),
        "RF_RMSE": np.sqrt(mean_squared_error(y_test, pred_rf)),
        "XGB_MAE": mean_absolute_error(y_test, pred_xgb),
        "XGB_RMSE": np.sqrt(mean_squared_error(y_test, pred_xgb)),
    }

    print(f"\n--- {name} ahead ---")
    print("Random Forest  MAE:", results[name]["RF_MAE"], " RMSE:", results[name]["RF_RMSE"])
    print("XGBoost        MAE:", results[name]["XGB_MAE"], " RMSE:", results[name]["XGB_RMSE"])

# ============================================
# STEP 9: Save results to a CSV so you can show it in your report
# ============================================
results_df = pd.DataFrame(results).T
results_df.to_csv("data/ml/model_results.csv")
print("\nSaved results to data/ml/model_results.csv")