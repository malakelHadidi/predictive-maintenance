import pandas as pd

from src.config import RAW_DATA_DIR, RESULTS_DIR
from src.data_loader import load_cmapss_data
from src.feature_engineering import engineer_features
from src.preprocessing import load_scaler, scale_data
from src.predict import predict_rul
from src.maintenance import get_maintenance_decision


def run_pipeline():
    """
    Run the complete predictive maintenance pipeline.
    """

    print("=" * 50)
    print("PREDICTIVE MAINTENANCE PIPELINE")
    print("=" * 50)

    # --------------------------------------------------
    # 1. Load test data
    # --------------------------------------------------

    print("\n[1/6] Loading test data...")

    test_path = RAW_DATA_DIR / "test_FD001.txt"

    test_df = load_cmapss_data(
        test_path
    )

    print(
        f"Loaded {test_df['unit'].nunique()} engines."
    )


    # --------------------------------------------------
    # 2. Feature engineering
    # --------------------------------------------------

    print("\n[2/6] Engineering features...")

    test_features, selected_sensors = engineer_features(
        test_df,
        calculate_target=False
    )

    print(
        f"Generated {len(test_features.columns)} columns."
    )


    # --------------------------------------------------
    # 3. Load scaler and scale data
    # --------------------------------------------------

    print("\n[3/6] Scaling features...")

    scaler = load_scaler()

    test_scaled = scale_data(
        test_features,
        scaler
    )

    print("Features scaled successfully.")


    # --------------------------------------------------
    # 4. Generate RUL predictions
    # --------------------------------------------------

    print("\n[4/6] Predicting RUL...")

    predictions = predict_rul(
        test_scaled
    )

    print(
        f"Generated {len(predictions)} predictions."
    )


    # --------------------------------------------------
    # 5. Apply maintenance decisions
    # --------------------------------------------------

    print("\n[5/6] Generating maintenance decisions...")

    results = []

    for unit_id, predicted_rul in predictions.items():

        decision = get_maintenance_decision(
            predicted_rul
        )

        results.append({
            "unit": unit_id,
            "predicted_RUL": predicted_rul,
            "status": decision["status"],
            "action": decision["action"]
        })


    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "unit"
    ).reset_index(drop=True)


    # --------------------------------------------------
    # 6. Save results
    # --------------------------------------------------

    print("\n[6/6] Saving results...")

    output_path = (
        RESULTS_DIR /
        "pipeline_predictions.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Results saved to: {output_path}"
    )


    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE")
    print("=" * 50)

    print("\nMaintenance summary:")

    print(
        results_df["status"].value_counts()
    )

    return results_df


if __name__ == "__main__":
    run_pipeline()