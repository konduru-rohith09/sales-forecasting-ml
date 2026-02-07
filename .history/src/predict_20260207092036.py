import joblib
import pandas as pd
from pathlib import Path

MODEL_VERSION = "v1"
MODELS_DIR = Path("model")

MODEL_PATH = MODELS_DIR / f"sales_forecast_model_{MODEL_VERSION}.joblib"
SCALER_PATH = MODELS_DIR / f"scaler_{MODEL_VERSION}.pkl"
FEATURES_PATH = MODELS_DIR / f"feature_columns_{MODEL_VERSION}.joblib"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURES_PATH)


def predict(input_df: pd.DataFrame):
    """
    input_df: pandas DataFrame with raw input features
    """
    df = input_df[feature_columns]
    scaled_data = scaler.transform(df)
    prediction = model.predict(scaled_data)
    return float(prediction[0])
