import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Air Quality Predictor",
    page_icon="🌫",
    layout="centered"
)

st.title("🌫 Air Quality Prediction System")
st.markdown("### Predict PM2.5 using Environmental Factors")

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    return joblib.load("air_quality_model.pkl")

model = load_model()
st.success("✅ Model loaded successfully")

# 🔥 AUTO GET FEATURE NAMES
feature_names = model.feature_names_in_

# ==============================
# USER INPUT SECTION
# ==============================
st.header("📥 Enter Environmental Parameters")

input_data = {}

for feature in feature_names:
    input_data[feature] = st.number_input(
        f"Enter {feature}",
        min_value=0.0,
        value=10.0,
        step=0.1
    )

# ==============================
# PREDICTION
# ==============================
if st.button("🔮 Predict Pollution Level"):

    try:
        input_df = pd.DataFrame([input_data])

        # Ensure correct column order
        input_df = input_df[feature_names]

        prediction = model.predict(input_df)[0]

        st.success("✅ Prediction Complete")
        st.metric("Predicted PM2.5", f"{prediction:.2f}")

    except Exception as e:
        st.error(f"❌ Error: {e}")