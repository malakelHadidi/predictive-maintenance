import numpy as np
import pandas as pd

import json

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_loader import load_cmapss_data
from src.feature_engineering import engineer_features
from src.preprocessing import load_scaler, scale_data, get_feature_columns
from src.predict import predict_rul


# --------------------------------------------------
# 1. Load test data
# --------------------------------------------------

test_df = load_cmapss_data(
    "data/raw/test_FD001.txt"
)

print("Test data shape:")
print(test_df.shape)


# --------------------------------------------------
# 2. Feature engineering
# --------------------------------------------------

test_features, selected_sensors = engineer_features(
    test_df,
    calculate_target=False
)

print("\nEngineered test data shape:")
print(test_features.shape)

print("\nNumber of test engines:")
print(test_features["unit"].nunique())

print("\nSelected sensors:")
print(selected_sensors)


# --------------------------------------------------
# 3. Load the existing scaler
# --------------------------------------------------

scaler = load_scaler()

test_scaled = scale_data(
    test_features,
    scaler
)

print("\nTest data scaled successfully.")


# --------------------------------------------------
# 4. Check feature count
# --------------------------------------------------

feature_cols = get_feature_columns(
    test_scaled
)

print("\nNumber of model features:")
print(len(feature_cols))

if len(feature_cols) != 112:
    raise ValueError(
        f"Expected 112 features, but found {len(feature_cols)}."
    )


# --------------------------------------------------
# 5. Generate predictions
# --------------------------------------------------

predictions_dict = predict_rul(
    test_scaled
)

print("\nNumber of predictions:")
print(len(predictions_dict))


# --------------------------------------------------
# 6. Load official RUL values
# --------------------------------------------------

actual_rul = pd.read_csv(
    "data/raw/RUL_FD001.txt",
    header=None,
    names=["RUL"]
)

print("\nOfficial RUL shape:")
print(actual_rul.shape)


# --------------------------------------------------
# 7. Verify that all 100 engines were predicted
# --------------------------------------------------

expected_units = sorted(
    test_scaled["unit"].unique()
)

predicted_units = sorted(
    predictions_dict.keys()
)

if expected_units != predicted_units:
    raise ValueError(
        "The predicted engine IDs do not match "
        "the test engine IDs."
    )

if len(actual_rul) != len(expected_units):
    raise ValueError(
        "Number of official RUL values does not "
        "match number of test engines."
    )


# --------------------------------------------------
# 8. Align predictions with engine IDs
# --------------------------------------------------

predicted_rul = np.array([
    predictions_dict[unit_id]
    for unit_id in expected_units
])

actual_rul_values = actual_rul["RUL"].to_numpy()


# --------------------------------------------------
# 9. Calculate metrics
# --------------------------------------------------

mae = mean_absolute_error(
    actual_rul_values,
    predicted_rul
)

rmse = np.sqrt(
    mean_squared_error(
        actual_rul_values,
        predicted_rul
    )
)

r2 = r2_score(
    actual_rul_values,
    predicted_rul
)


# --------------------------------------------------
# 10. Display final results
# --------------------------------------------------

print("\n" + "=" * 50)
print("OFFICIAL TEST RESULTS")
print("=" * 50)

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")


# --------------------------------------------------
# 11. Create prediction results table
# --------------------------------------------------

results_df = pd.DataFrame({
    "unit": expected_units,
    "actual_RUL": actual_rul_values,
    "predicted_RUL": predicted_rul
})

print("\nFirst 10 predictions:")
print(results_df.head(10))


# --------------------------------------------------
# 12. Save predictions
# --------------------------------------------------

results_df.to_csv(
    "results/test_predictions.csv",
    index=False
)

print(
    "\nPredictions saved to "
    "results/test_predictions.csv"
)

metrics = {
    "MAE": float(mae),
    "RMSE": float(rmse),
    "R2": float(r2)
}

with open(
    "results/final_metrics.json",
    "w"
) as f:
    json.dump(
        metrics,
        f,
        indent=4
    )

print(
    "Metrics saved to "
    "results/final_metrics.json"
)