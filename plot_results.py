import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Load official predictions
# --------------------------------------------------

results_df = pd.read_csv(
    "results/test_predictions.csv"
)


# --------------------------------------------------
# 2. Calculate prediction error
# --------------------------------------------------

results_df["error"] = (
    results_df["predicted_RUL"]
    - results_df["actual_RUL"]
)


# --------------------------------------------------
# 3. Actual vs Predicted RUL
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    results_df["actual_RUL"],
    results_df["predicted_RUL"]
)

# Perfect prediction line
min_rul = min(
    results_df["actual_RUL"].min(),
    results_df["predicted_RUL"].min()
)

max_rul = max(
    results_df["actual_RUL"].max(),
    results_df["predicted_RUL"].max()
)

plt.plot(
    [min_rul, max_rul],
    [min_rul, max_rul],
    linestyle="--"
)

plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")
plt.title("Actual vs Predicted RUL")

plt.tight_layout()

plt.savefig(
    "results/actual_vs_predicted_rul.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 4. Prediction Error by Engine
# --------------------------------------------------

plt.figure(figsize=(12, 6))

plt.bar(
    results_df["unit"],
    results_df["error"]
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Engine")
plt.ylabel("Prediction Error (Predicted - Actual)")
plt.title("RUL Prediction Error by Engine")

plt.tight_layout()

plt.savefig(
    "results/prediction_error_by_engine.png",
    dpi=300
)

plt.show()


print(
    "\nFigures saved successfully:"
)

print(
    "results/actual_vs_predicted_rul.png"
)

print(
    "results/prediction_error_by_engine.png"
)