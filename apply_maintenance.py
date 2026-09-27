import pandas as pd

from src.maintenance import get_maintenance_decision


# --------------------------------------------------
# 1. Load model predictions
# --------------------------------------------------

results_df = pd.read_csv(
    "results/test_predictions.csv"
)


# --------------------------------------------------
# 2. Apply maintenance decision
# --------------------------------------------------

decisions = results_df["predicted_RUL"].apply(
    get_maintenance_decision
)


# --------------------------------------------------
# 3. Add decision information
# --------------------------------------------------

results_df["status"] = decisions.apply(
    lambda x: x["status"]
)

results_df["action"] = decisions.apply(
    lambda x: x["action"]
)


# --------------------------------------------------
# 4. Display results
# --------------------------------------------------

print("\nMaintenance decisions:")
print(
    results_df[
        [
            "unit",
            "predicted_RUL",
            "status",
            "action"
        ]
    ].head(10)
)


# --------------------------------------------------
# 5. Count each maintenance status
# --------------------------------------------------

print("\nMaintenance status counts:")

print(
    results_df["status"].value_counts()
)


# --------------------------------------------------
# 6. Save results
# --------------------------------------------------

results_df.to_csv(
    "results/maintenance_decisions.csv",
    index=False
)

print(
    "\nMaintenance decisions saved to "
    "results/maintenance_decisions.csv"
)