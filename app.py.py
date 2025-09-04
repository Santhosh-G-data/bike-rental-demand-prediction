# AI Bike Rentals - PUBG-Style Futuristic UI

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import json
import os

st.set_page_config(page_title="AI Bike Rentals", layout="wide")

# ----------------------------- STYLES ---------------------------------- #
lottie_path = os.path.join(os.path.dirname(__file__), "lotties", "bike.json")
print("Looking for Lottie file at:", lottie_path)  # 🐞 Debug print

with open(lottie_path, "r") as f:
    bike_lottie = json.load(f)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Orbitron', sans-serif;
        background-color: #0c0f1a;
        color: #e0e0e0;
    }

    .title {
        text-align: center;
        font-size: 48px;
        color: #ffe100;
        text-shadow: 0 0 10px #ffe100;
        margin-top: 20px;
    }

    .section {
        border: 1px solid #2a2e35;
        border-radius: 16px;
        padding: 24px;
        background-color: #1a1d29;
        box-shadow: 0 0 8px #00ffff40;
    }

    .predict-btn button {
        background-color: #ff4545 !important;
        color: white;
        font-size: 18px;
        border-radius: 12px;
        width: 100%;
        padding: 10px;
        box-shadow: 0 0 10px red;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🚲 AI BIKE RENTALS PREDICTOR 🔮</div>", unsafe_allow_html=True)

# ---------------------------- SIDEBAR ---------------------------------- #
# st.sidebar.image("images/logo.png", use_column_width=True)
st.sidebar.header("🔧 Input Configuration")

def user_input():
    season = st.sidebar.selectbox('Season', [1, 2, 3, 4])
    yr = st.sidebar.selectbox('Year', [0, 1])
    mnth = st.sidebar.slider('Month', 1, 12, 6)
    hr = st.sidebar.slider('Hour', 0, 23, 10)
    holiday = st.sidebar.selectbox('Holiday', [0, 1])
    weekday = st.sidebar.slider('Weekday', 0, 6, 2)
    workingday = st.sidebar.selectbox('Working Day', [0, 1])
    weathersit = st.sidebar.slider('Weather Situation', 1, 4, 2)
    temp = st.sidebar.slider('Temperature', 0.0, 1.0, 0.5)
    atemp = st.sidebar.slider('Feeling Temperature', 0.0, 1.0, 0.5)
    hum = st.sidebar.slider('Humidity', 0.0, 1.0, 0.4)
    windspeed = st.sidebar.slider('Windspeed', 0.0, 1.0, 0.3)

    data = {
        'season': season,
        'yr': yr,
        'mnth': mnth,
        'hr': hr,
        'holiday': holiday,
        'weekday': weekday,
        'workingday': workingday,
        'weathersit': weathersit,
        'temp': temp,
        'atemp': atemp,
        'hum': hum,
        'windspeed': windspeed
    }
    return pd.DataFrame([data])

input_df = user_input()

# ---------------------------- MODEL ZONE ------------------------------- #
# Load training data
bike_data = pd.read_csv("day.csv")
X = bike_data.drop(['cnt', 'dteday', 'casual', 'registered'], axis=1)
y = bike_data['cnt']

model = XGBRegressor(n_estimators=2000, learning_rate=0.005, max_depth=5,
                     min_child_weight=6, subsample=0.7, colsample_bytree=0.7,
                     max_leaves=16, reg_lambda=2.0, random_state=42)
model.fit(X, y)

prediction = model.predict(input_df)

# ---------------------------- MAIN PANEL ------------------------------- #
col1, col2 = st.columns([2, 3])

with col1:
    st_lottie(bike_lottie, speed=1.2, height=300)
    st.markdown(f"""
        <div class="section">
            <h3 style='color:#00ffff'>🎯 PREDICTION OUTPUT</h3>
            <p style='font-size:28px; color:#ffe100;'>Predicted Rentals: <strong>{prediction[0]:.0f}</strong></p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="section">
            <h3 style='color:#00ffff'>📊 PERFORMANCE RADAR</h3>
    """, unsafe_allow_html=True)

    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(
        r=[input_df.temp[0], input_df.atemp[0], input_df.hum[0], input_df.windspeed[0]],
        theta=['Temp', 'Feeling Temp', 'Humidity', 'Windspeed'],
        fill='toself',
        name='User Input'
    ))
    radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        showlegend=False,
                        template='plotly_dark')
    st.plotly_chart(radar, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------- FOOTER ------------------------------- #
st.markdown("""
    <hr style='border-color:#2a2e35;'>
    <center style='color:#888;'>
        🚀 Powered by Advanced AI • Styled after PUBG Predictor UI • Built with ❤️ using Streamlit
    </center>
""", unsafe_allow_html=True)
