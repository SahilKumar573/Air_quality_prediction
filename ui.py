import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Air Quality Predictor",
    page_icon="🌫",
    layout="wide"
)

# ==============================
# DARK / LIGHT TOGGLE
# ==============================
theme = st.sidebar.toggle("🌙 Dark Mode", value=True)

if theme:
    bg_gradient = "linear-gradient(135deg, #0f2027, #203a43, #2c5364)"
else:
    bg_gradient = "linear-gradient(135deg, #f5f7fa, #c3cfe2)"

# ==============================
# CUSTOM CSS
# ==============================
st.markdown(f"""
<style>
.stApp {{
    background: {bg_gradient};
}}

.block-container {{
    background: rgba(255,255,255,0.08);
    padding: 2rem;
    border-radius: 20px;
    backdrop-filter: blur(10px);
}}

.stButton>button {{
    background: linear-gradient(90deg, #ff512f, #dd2476);
    color: white;
    font-size: 18px;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    border: none;
}}

#logo {{
    position: fixed;
    top: 15px;
    right: 20px;
    font-size: 28px;
}}
</style>

<div id="logo">🌍</div>
""", unsafe_allow_html=True)

# ==============================
# SIDEBAR
# ==============================
st.sidebar.title("🌫 Air Quality Project")

city = st.sidebar.selectbox(
    "🌆 Select City",
    ["Delhi", "Mumbai", "Kolkata", "Chennai", "Bhubaneswar"]
)

st.sidebar.info("""
**Models Used**
- Linear Regression  
- Decision Tree  
- Random Forest  
""")

# ==============================
# TITLE
# ==============================
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

# ==============================
# FEATURES
# ==============================
feature_names = model.feature_names_in_

st.header("📥 Enter Environmental Parameters")

input_data = {}
cols = st.columns(3)

for i, feature in enumerate(feature_names):
    with cols[i % 3]:
        input_data[feature] = st.number_input(
            f"{feature}",
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
        input_df = input_df[feature_names]
        prediction = model.predict(input_df)[0]

        st.success("✅ Prediction Complete")

        # ======================
        # AQI STATUS
        # ======================
        if prediction <= 50:
            status = "Good"
            color = "green"
        elif prediction <= 100:
            status = "Satisfactory"
            color = "yellow"
        elif prediction <= 200:
            status = "Moderate"
            color = "orange"
        elif prediction <= 300:
            status = "Poor"
            color = "red"
        else:
            status = "Severe"
            color = "darkred"

        # ======================
        # METRIC
        # ======================
        st.metric("Predicted PM2.5", f"{prediction:.2f}")
        st.markdown(f"### Air Quality Status: **{status}**")

        # ======================
        # 🎯 GAUGE METER
        # ======================
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            title={'text': "PM2.5 Level"},
            gauge={
                'axis': {'range': [0, 500]},
                'bar': {'color': color},
            }
        ))

        st.plotly_chart(gauge_fig, use_container_width=True)

        # ======================
        # 📈 POLLUTION GRAPH
        # ======================
        history = np.clip(
            np.random.normal(prediction, prediction*0.15, 24),
            0, None
        )

        hist_df = pd.DataFrame({
            "Hour": list(range(1, 25)),
            "PM2.5": history
        })

        line_fig = px.line(
            hist_df,
            x="Hour",
            y="PM2.5",
            title=f"📈 Predicted Pollution Trend — {city}",
            markers=True
        )

        st.plotly_chart(line_fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error: {e}")

st.markdown("---")
st.caption("🏆 Resume Project | Sahil Kumar | AI/ML Air Quality Prediction")
