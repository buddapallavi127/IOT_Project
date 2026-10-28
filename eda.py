import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/ml/traffic_long.csv")

# Plot 1: average speed over time, one line per scenario
plt.figure(figsize=(10,5))
for scenario in df.scenario.unique():
    sub = df[df.scenario == scenario]
    avg_speed = sub.groupby("time")["speed_ms"].mean()
    plt.plot(avg_speed.index, avg_speed.values, label=scenario)
plt.xlabel("Time (s)")
plt.ylabel("Avg speed (m/s)")
plt.title("Average speed over time by scenario")
plt.legend()
plt.savefig("data/ml/speed_over_time.png")
plt.show()

# Plot 2: congestion rate per scenario (bar chart)
plt.figure(figsize=(8,5))
df.groupby("scenario")["congested"].mean().plot(kind="bar")
plt.ylabel("Fraction of time congested")
plt.title("Congestion rate by scenario")
plt.tight_layout()
plt.savefig("data/ml/congestion_by_scenario.png")
plt.show()

print("Saved plots to data/ml/")