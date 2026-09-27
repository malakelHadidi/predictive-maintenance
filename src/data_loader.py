import pandas as pd


# C-MAPSS column names
COLUMN_NAMES = [
    "unit",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21"
]


def load_cmapss_data(file_path):
    """
    Load a C-MAPSS dataset file.

    Parameters
    ----------
    file_path : str or Path
        Path to the C-MAPSS text file.

    Returns
    -------
    pandas.DataFrame
        Dataset with properly assigned column names.
    """

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None
    )

    df.columns = COLUMN_NAMES

    return df