from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"


MODEL_PATH = MODEL_DIR / "stacked_lstm_rul.keras"

SCALER_PATH = MODEL_DIR / "lstm_feature_scaler.joblib"

SEQUENCE_LENGTH = 30

SELECTED_SENSORS = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21"
]

EXCLUDE_COLUMNS = [
    "unit",
    "cycle",
    "RUL",
    "setting_1",
    "setting_2",
    "setting_3"
]