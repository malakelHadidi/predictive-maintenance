
import numpy as np

from src.config import SEQUENCE_LENGTH
from src.preprocessing import get_feature_columns
from src.model_loader import load_lstm_model


def predict_rul(df):
    """
    Generate one RUL prediction for each engine.

    The prediction is based on the final available
    sequence of SEQUENCE_LENGTH cycles for each engine.

    Parameters
    ----------
    df : pandas.DataFrame
        Engineered and scaled dataframe.

    Returns
    -------
    predictions : dict
        Dictionary mapping engine ID to predicted RUL.
    """

    feature_cols = get_feature_columns(df)

    predictions = {}

    model = load_lstm_model()

    for unit_id in df["unit"].unique():

        unit_df = (
            df[df["unit"] == unit_id]
            .sort_values("cycle")
        )

        if len(unit_df) < SEQUENCE_LENGTH:
            continue

        final_sequence = (
            unit_df[feature_cols]
            .values[-SEQUENCE_LENGTH:]
        )

        X = np.expand_dims(
            final_sequence,
            axis=0
        )

        prediction = model.predict(
            X,
            verbose=0
        )[0][0]

        predictions[unit_id] = float(prediction)

    return predictions


def classify_rul(predicted_rul):
    """
    Convert predicted RUL into a maintenance status.
    """

    if predicted_rul <= 10:
        return "CRITICAL"

    elif predicted_rul <= 30:
        return "WARNING"

    else:
        return "NORMAL"
