from fastapi import FastAPI
from app.schema import WeatherInput
from app.utils import predict

# Initialize API
app = FastAPI(title="Weather Classification API")


@app.get("/")
def home():
    return {"message": "Weather Classification API is running"}


@app.post("/predict")
def predict_weather(data: WeatherInput):
    # Convert Pydantic object to dict
    input_dict = {
        "Temperature": data.Temperature,
        "Humidity": data.Humidity,
        "Wind Speed": data.Wind_Speed,
        "Precipitation (%)": data.Precipitation,
        "Cloud Cover": data.Cloud_Cover,
        "Atmospheric Pressure": data.Atmospheric_Pressure,
        "UV Index": data.UV_Index,
        "Season": data.Season,
        "Visibility (km)": data.Visibility,
        "Location": data.Location
    }

    result = predict(input_dict)

    return {"prediction": result}