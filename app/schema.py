from pydantic import BaseModel

# Input schema for API
class WeatherInput(BaseModel):
    Temperature: float
    Humidity: float
    Wind_Speed: float
    Precipitation: float
    Cloud_Cover: str
    Atmospheric_Pressure: float
    UV_Index: float
    Season: str
    Visibility: float
    Location: str