import joblib 
import pandas as pd 

MODEL_VERSION="v1"
MODEL_PATH=f"model/sales_forecast_model_{MODEL_VERSION}.joblib"
SCALAR_PATH=f"model/scaler_{MODEL_VERSION}.pkl"
FEATURE_PATH=f"model/feature_{MODEL_VERSION}.joblib"
model=joblib.load(MODEL_PATH)
feature_columns=joblib.load(FEATURE_PATH)
scalar=jo
