import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import datetime


MODELS_DIR = Path("../model")


# ---------- Load Model ----------
@st.cache_resource
def load_model():
    model = joblib.load(MODELS_DIR / "sales_forecast_model_v1.joblib")
    feature_cols = joblib.load(MODELS_DIR / "feature_columns-.joblib")
    return model, feature_cols


model, feature_cols = load_model()


# ---------- UI ----------
st.set_page_config(
    page_title="Retail Sales Revenue Forecasting",
    page_icon="📈",
    layout="centered"
)

st.title("📊 Sales Revenue Forecasting Dashboard")
st.write("Forecast expected sales for a retail store using ML")


# ---------- Sidebar Inputs ----------
st.sidebar.header("Store & Promotion Inputs")

store_id = st.sidebar.number_input(
    "Store ID", min_value=1, max_value=2000, value=1
)

promo = st.sidebar.selectbox(
    "Promo Running?",
    options=[0, 1],
    index=0
)

promo2 = st.sidebar.selectbox(
    "Long-Term Promo (Promo2)?",
    options=[0, 1],
    index=0
)

competition_distance = st.sidebar.number_input(
    "Competition Distance (meters)",
    min_value=0,
    max_value=50000,
    value=1000
)

day_of_week = st.sidebar.selectbox(
    "Day of Week (0=Mon ... 6=Sun)",
    options=list(range(7)),
    index=0
)

is_weekend = 1 if day_of_week in [5, 6] else 0

week_of_year = datetime.datetime.now().isocalendar()[1]
quarter = (datetime.datetime.now().month - 1) // 3 + 1


# ---------- Lag & Rolling Inputs ----------
st.sidebar.header("Lag Features")

lag1 = st.sidebar.number_input("Sales Lag 1 Day", value=5000)
lag7 = st.sidebar.number_input("Sales Lag 7 Days", value=5200)
lag30 = st.sidebar.number_input("Sales Lag 30 Days", value=5100)

roll7 = st.sidebar.number_input("Rolling 7-Day Avg", value=5150)
roll30 = st.sidebar.number_input("Rolling 30-Day Avg", value=5050)


# ---------- Build Input Row ----------
input_data = pd.DataFrame([[
    store_id,
    promo,
    promo2,
    competition_distance,
    1 if competition_distance < 5000 else 0,
    day_of_week,
    is_weekend,
    week_of_year,
    quarter,
    lag1,
    lag7,
    lag30,
    roll7,
    roll30
]], columns=feature_cols)


# ---------- Predict ----------
if st.button("Predict Sales"):

    prediction = model.predict(input_data)[0]

    st.subheader("📌 Forecast Result")
    st.metric(
        label="Predicted Daily Sales",
        value=f"€ {prediction:,.2f}"
    )

    st.write("---")
    st.write("### 🧠 Model Inputs Used")
    st.write(input_data)

else:
    st.info("Click **Predict Sales** to generate forecast ⬆️")
