import streamlit as st
import joblib
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

model = joblib.load("holiday_predictor.pkl")
LAT, LON = 10.5276, 76.2144
API_URL = "https://api.open-meteo.com/v1/forecast"

st.title("Will Thrissur Collector give holiday Tommorow?")
st.write("Predict whether the Collector will announce a holiday tomorrow based on weather data.")

if st.button("Predict for Tomorrow"):
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    tomorrow = today + timedelta(days=1)

    dates = [yesterday, today, tomorrow]
    date_strs = [d.isoformat() for d in dates]
    params = {
        "latitude": LAT,
        "longitude": LON,
        "daily": "precipitation_sum,rain_sum,temperature_2m_max,temperature_2m_min",
        "start_date": date_strs[0],
        "end_date": date_strs[2],
        "timezone": "Asia/Kolkata"
    }

    response = requests.get(API_URL, params=params)
    resp = response.json()
    df = pd.DataFrame(resp["daily"])

    features = [
        df["precipitation_sum"][1],  # today
        df["rain_sum"][1],
        df["temperature_2m_max"][1],
        df["temperature_2m_min"][1],
         df["precipitation_sum"][2], #tmrw
        df["rain_sum"][2],           
        df["temperature_2m_max"][2],
        df["temperature_2m_min"][2],
        df["precipitation_sum"][0],  #prev
        df["rain_sum"][0],
        df["temperature_2m_max"][0],
        df["temperature_2m_min"][0]
    ]

    input_arr = np.array([features])
    pred = model.predict(input_arr)[0]

    if pred == 1:
        st.confetti("Yay! Naale Modak illa..")
        st.balloons()
        st.image(".\happy-salim-kumar.gif",use_container_width=True)
    else:
        st.warning("❌ Naale Modak illa.. Schoolil poyittu vaaa..")
        st.image(".\mammootty-vipin-ayilam.gif",use_container_width=True)


