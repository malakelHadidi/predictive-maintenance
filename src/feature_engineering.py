import pandas as pd
from src.config import SELECTED_SENSORS

def calculate_rul(df):
    """
    Calculate Remaining Useful Life (RUL) for each engine.

    RUL = maximum cycle of the engine - current cycle.
    """

    df = df.copy()

    max_cycles = (
        df.groupby("unit")["cycle"]
        .transform("max")
    )

    df["RUL"] = max_cycles - df["cycle"]

    return df


def select_sensors(df):
    """
    Return the exact sensors used by the trained model.
    """

    return SELECTED_SENSORS.copy()

def prepare_base_data(df, selected_sensors):
    """
    Keep only the columns used by the feature-engineering pipeline.
    """

    columns = (
        [
            "unit",
            "cycle",
            "setting_1",
            "setting_2",
            "setting_3"
        ]
        + selected_sensors
        + ["RUL"]
    )

    df = df[columns].copy()

    df = (
        df.sort_values(["unit", "cycle"])
        .reset_index(drop=True)
    )

    return df


def add_rolling_mean_features(df, selected_sensors, windows=None):
    """
    Add rolling mean features for each selected sensor.
    """

    if windows is None:
        windows = [5, 10, 20]

    rolling_mean_features = {}

    for window in windows:
        for sensor in selected_sensors:

            rolling_mean_features[
                f"{sensor}_roll_mean_{window}"
            ] = (
                df
                .groupby("unit")[sensor]
                .transform(
                    lambda x: x.rolling(
                        window=window,
                        min_periods=1
                    ).mean()
                )
            )

    rolling_mean_df = pd.DataFrame(
        rolling_mean_features,
        index=df.index
    )

    df = pd.concat(
        [df, rolling_mean_df],
        axis=1
    )

    return df


def add_rolling_std_features(df, selected_sensors, windows=None):
    """
    Add rolling standard deviation features for each selected sensor.
    """

    if windows is None:
        windows = [5, 10, 20]

    rolling_std_features = {}

    for window in windows:
        for sensor in selected_sensors:

            rolling_std_features[
                f"{sensor}_roll_std_{window}"
            ] = (
                df
                .groupby("unit")[sensor]
                .transform(
                    lambda x: x.rolling(
                        window=window,
                        min_periods=2
                    ).std()
                )
            )

    rolling_std_df = pd.DataFrame(
        rolling_std_features,
        index=df.index
    )

    df = pd.concat(
        [df, rolling_std_df],
        axis=1
    )

    roll_std_cols = [
        col
        for col in df.columns
        if "_roll_std_" in col
    ]

    df[roll_std_cols] = (
        df[roll_std_cols]
        .fillna(0)
    )

    return df


def add_difference_features(df, selected_sensors):
    """
    Add cycle-to-cycle difference features.

    Difference = current sensor value - previous sensor value
    within the same engine.
    """

    diff_features = {}

    for sensor in selected_sensors:

        diff_features[
            f"{sensor}_diff"
        ] = (
            df
            .groupby("unit")[sensor]
            .diff()
        )

    diff_df = pd.DataFrame(
        diff_features,
        index=df.index
    )

    df = pd.concat(
        [df, diff_df],
        axis=1
    )

    diff_cols = [
        col
        for col in df.columns
        if col.endswith("_diff")
    ]

    df[diff_cols] = (
        df[diff_cols]
        .fillna(0)
    )

    return df


def engineer_features(df, calculate_target=False):
    """
    Complete feature-engineering pipeline.

    Parameters
    ----------
    df : pandas.DataFrame
        C-MAPSS dataframe.

    calculate_target : bool
        If True, calculate RUL.
        Use True for training data.
        Use False for test/inference data.

    Returns
    -------
    df : pandas.DataFrame
        Engineered dataframe.

    selected_sensors : list
        Sensors retained after variance filtering.
    """

    df = df.copy()

    # Calculate RUL only when requested
    if calculate_target:
        df = calculate_rul(df)

    # Select useful sensors
    selected_sensors = select_sensors(df)

    # Build the base dataframe
    columns = (
        [
            "unit",
            "cycle",
            "setting_1",
            "setting_2",
            "setting_3"
        ]
        + selected_sensors
    )

    if calculate_target:
        columns.append("RUL")

    df = df[columns].copy()

    # Sort chronologically within each engine
    df = (
        df.sort_values(["unit", "cycle"])
        .reset_index(drop=True)
    )

    # Add rolling means
    df = add_rolling_mean_features(
        df,
        selected_sensors
    )

    # Add rolling standard deviations
    df = add_rolling_std_features(
        df,
        selected_sensors
    )

    # Add sensor differences
    df = add_difference_features(
        df,
        selected_sensors
    )

    return df, selected_sensors