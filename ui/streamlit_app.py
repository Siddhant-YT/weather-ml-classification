import streamlit as st
import requests, os
import pandas as pd
import plotly.express as px

# API URL
API_URL = "http://127.0.0.1:8000/predict"

st.title("Weather Classification System")

st.write("""
This application predicts the weather type based on environmental conditions.
The model is trained using multiple classification algorithms and optimized using hyperparameter tuning.
""")

# --------------------------
# INPUT FORM
# --------------------------
st.header("Enter Weather Parameters")

temperature = st.number_input("Temperature")
humidity = st.number_input("Humidity")
wind_speed = st.number_input("Wind Speed")
precipitation = st.number_input("Precipitation (%)")
cloud_cover = st.selectbox("Cloud Cover", ["overcast", "partly cloudy", "clear", "cloudy"])
pressure = st.number_input("Atmospheric Pressure")
uv_index = st.number_input("UV Index")
season = st.selectbox("Season", ["Summer", "Winter", "Spring", "Autumn"])
visibility = st.number_input("Visibility (km)")
location = st.selectbox("Location", ["inland", "coastal", "mountain"])

# --------------------------
# PREDICT BUTTON
# --------------------------
if st.button("Predict Weather"):
    payload = {
        "Temperature": temperature,
        "Humidity": humidity,
        "Wind_Speed": wind_speed,
        "Precipitation": precipitation,
        "Cloud_Cover": cloud_cover,
        "Atmospheric_Pressure": pressure,
        "UV_Index": uv_index,
        "Season": season,
        "Visibility": visibility,
        "Location": location
    }

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        prediction = response.json()["prediction"]
        st.success(f"Predicted Weather Type: {prediction}")
    else:
        st.error("Error in prediction")

# # --------------------------
# # DATA VISUALIZATION
# # --------------------------
# st.header("Dataset Insights")

# # Load dataset
# # df = pd.read_csv("../data/weather_classification_data.csv")
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # points to 4_Classification_ML/
# df = pd.read_csv(os.path.join(BASE_DIR, "data", "weather_classification_data.csv"))


# # Interactive histogram
# fig1 = px.histogram(df, x="Temperature", color="Weather Type",
#                     title="Temperature Distribution by Weather Type")
# st.plotly_chart(fig1)

# # Scatter plot
# fig2 = px.scatter(df, x="Humidity", y="Temperature",
#                   color="Weather Type",
#                   title="Humidity vs Temperature")
# st.plotly_chart(fig2)

# # Box plot
# fig3 = px.box(df, x="Weather Type", y="Precipitation (%)",
#               title="Precipitation Distribution")
# st.plotly_chart(fig3)