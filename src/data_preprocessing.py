import pandas as pd
import numpy as np
from pathlib import Path


RAW_DATA_DIR = Path("../data/raw")
PROCESSED_DIR = Path("../data/processed")


def load_data():
    train = pd.read_csv(RAW_DATA_DIR / "train.csv")
    store = pd.read_csv(RAW_DATA_DIR / "store.csv")
    df = pd.merge(train, store, on="Store", how="left")
    return df


def clean_data(df):
    df = df[df["Open"] == 1]
    df = df[df["Sales"] > 0]
    df["Date"] = pd.to_datetime(df["Date"])
    return df.reset_index(drop=True)


def add_features(df):
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["IsWeekend"] = df["DayOfWeek"].isin([5,6]).astype(int)
    df["Quarter"] = df["Date"].dt.quarter

    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(
        df["CompetitionDistance"].median()
    )
    df["HasCompetition"] = (df["CompetitionDistance"] < 5000).astype(int)

    df = df.sort_values(by=["Store","Date"])

    df["Sales_Lag_1"] = df.groupby("Store")["Sales"].shift(1)
    df["Sales_Lag_7"] = df.groupby("Store")["Sales"].shift(7)
    df["Sales_Lag_30"] = df.groupby("Store")["Sales"].shift(30)

    df["Sales_Rolling_7"] = df.groupby("Store")["Sales"].shift(1).rolling(7).mean()
    df["Sales_Rolling_30"] = df.groupby("Store")["Sales"].shift(1).rolling(30).mean()

    df = df.dropna().reset_index(drop=True)

    return df


def save_processed(df):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / "model_input.csv", index=False)
    print("Saved → data/processed/model_input.csv")


def main():
    print("Loading data...")
    df = load_data()

    print("Cleaning...")
    df = clean_data(df)

    print("Engineering features...")
    df = add_features(df)

    print("Saving...")
    save_processed(df)

    print("Done!")


if __name__ == "__main__":
    main()
