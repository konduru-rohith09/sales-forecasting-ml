import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import lightgbm as lgb
import joblib
mlflow.set()


PROCESSED_DIR = Path("../data/processed")
MODELS_DIR = Path("../models")


def load_data():
    df = pd.read_csv(PROCESSED_DIR / "model_input.csv")
    return df


def split_data(df):
    X = df.drop("Sales", axis=1)
    y = df["Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape


def train_models(X_train, y_train, X_test, y_test):
    results = {}

    # ---------- Linear Regression ----------
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    results["linear_regression"] = evaluate_model(y_test, y_pred_lr)

    # ---------- Random Forest ----------
    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results["random_forest"] = evaluate_model(y_test, y_pred_rf)

    # ---------- LightGBM ----------
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbose": -1,
    }

    lgb_model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        num_boost_round=500,
        early_stopping_rounds=50,
        verbose_eval=False
    )

    y_pred_lgb = lgb_model.predict(X_test)
    results["lightgbm"] = evaluate_model(y_test, y_pred_lgb)

    return results, lr, rf, lgb_model


def save_best_model(results, models, feature_columns):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Lower RMSE is better
    best_model_name = min(results, key=lambda x: results[x][1])
    best_model = models[best_model_name]

    joblib.dump(best_model, MODELS_DIR / "sales_forecast_model.joblib")
    joblib.dump(feature_columns, MODELS_DIR / "feature_columns.joblib")

    print(f"Saved best model: {best_model_name}")
    print("Metrics (MAE, RMSE, MAPE):", results[best_model_name])


def main():
    print("Loading data...")
    df = load_data()

    X_train, X_test, y_train, y_test = split_data(df)

    print("Training models...")
    results, lr, rf, lgb_model = train_models(X_train, y_train, X_test, y_test)

    models = {
        "linear_regression": lr,
        "random_forest": rf,
        "lightgbm": lgb_model
    }

    print("Saving best model...")
    save_best_model(results, models, list(X_train.columns))

    print("Done!")


if __name__ == "__main__":
    main()
