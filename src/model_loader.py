from tensorflow.keras.models import load_model

from src.config import MODEL_PATH


def load_lstm_model():
    """
    Load the trained stacked LSTM RUL model.
    """

    model = load_model(MODEL_PATH)

    return model