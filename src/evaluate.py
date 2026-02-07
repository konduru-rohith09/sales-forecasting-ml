import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROCESSED_DIR = Path("../data/processed")
MODELS_DIR = Path("../models")
REPORTS_DIR = Path("../reports")


def load_data():
    df = pd.read_csv(PROCESSED_DIR / "model_input.csv")
    return df


def load_model():
    model = joblib.load(MODELS_DIR / "sales_forecast_model_v1.joblib")
    feature_cols = joblib.load(MODELS_DIR / "feature_columns_v1.joblib")
    return model, feature_cols


def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape


def save_report(metrics):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df_metrics = pd.DataFrame([metrics])
    df_metrics.to_csv(REPORTS_DIR / "evaluation_metrics.csv", index=False)
    print("Saved → reports/evaluation_metrics.csv")


def main():
    print("Loading data...")
    df = load_data()

    print("Loading model...")
    model, feature_cols = load_model()

    print("Generating predictions...")
    X = df[feature_cols]
    y = df["Sales"]
    y_pred = model.predict(X)

    print("Evaluating...")
    mae, rmse, mape = evaluate_model(y, y_pred)

    metrics = {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE_percent": round(mape, 2)
    }

    print("\n--- Evaluation Report ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\nSaving report...")
    save_report(metrics)

    print("\nDone!")


if __name__ == "__main__":
    main()
