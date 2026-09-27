import joblib

from sklearn.preprocessing import StandardScaler

from src.config import EXCLUDE_COLUMNS, SCALER_PATH


def get_feature_columns(df):
    """
    Return the feature columns used by the final LSTM.

    Excludes:
        - unit
        - cycle
        - RUL
        - operating settings
    """

    feature_cols = [
        col
        for col in df.columns
        if col not in EXCLUDE_COLUMNS
    ]

    return feature_cols


def scale_data(df, scaler):
    """
    Scale feature columns using an already-fitted scaler.

    The scaler must already have been fitted on the
    appropriate training data.
    """

    feature_cols = get_feature_columns(df)

    scaled_df = df.copy()

    scaled_df[feature_cols] = scaler.transform(
        scaled_df[feature_cols]
    )

    return scaled_df


def load_scaler():
    """
    Load the scaler used by the final LSTM.
    """

    return joblib.load(SCALER_PATH)