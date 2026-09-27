import numpy as np

from src.config import SEQUENCE_LENGTH


def create_sequences(
    df,
    feature_cols,
    sequence_length=SEQUENCE_LENGTH,
    include_target=True
):
    """
    Create LSTM sequences from engine time-series data.

    Parameters
    ----------
    df : pandas.DataFrame
        Scaled engine dataframe.

    feature_cols : list
        Features used by the LSTM.

    sequence_length : int
        Number of cycles in each sequence.

    include_target : bool
        If True, return RUL targets.
        If False, only return input sequences.

    Returns
    -------
    X : numpy.ndarray
        LSTM input sequences.

    y : numpy.ndarray or None
        RUL targets when include_target=True.
    """

    X = []
    y = []

    for unit_id in df["unit"].unique():

        unit_df = (
            df[df["unit"] == unit_id]
            .sort_values("cycle")
        )

        feature_values = (
            unit_df[feature_cols].values
        )

        if include_target:
            rul_values = (
                unit_df["RUL"].values
            )

        for i in range(
            len(unit_df) - sequence_length + 1
        ):

            X.append(
                feature_values[
                    i:i + sequence_length
                ]
            )

            if include_target:
                y.append(
                    rul_values[
                        i + sequence_length - 1
                    ]
                )

    X = np.array(X)

    if include_target:
        return X, np.array(y)

    return X