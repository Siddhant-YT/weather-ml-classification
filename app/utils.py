import pandas as pd
import joblib, os

# # Load artifacts once (efficient)
# model = joblib.load("../models/rf_weather_model.pkl")
# scaler = joblib.load("../models/scaler.pkl")
# columns = joblib.load("../models/columns.pkl")
# outlier_bounds = joblib.load("../models/outlier_bounds.pkl")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # points to 4_Classification_ML/

model = joblib.load(os.path.join(BASE_DIR, "models", "rf_weather_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
columns = joblib.load(os.path.join(BASE_DIR, "models", "columns.pkl"))
outlier_bounds = joblib.load(os.path.join(BASE_DIR, "models", "outlier_bounds.pkl"))
# Numerical columns
num_cols = ['Temperature', 'Humidity', 'Wind Speed',
            'Precipitation (%)', 'Atmospheric Pressure',
            'UV Index', 'Visibility (km)']


def preprocess_input(input_data: dict):
    # Convert to DataFrame
    df = pd.DataFrame([input_data])

    # Handle outliers using saved bounds
    for col in num_cols:
        lower, upper = outlier_bounds[col]
        df[col] = df[col].clip(lower, upper)

    # One-hot encoding
    df = pd.get_dummies(df)

    # Align columns with training data
    df = df.reindex(columns=columns, fill_value=0)

    # Scale numerical features
    df[num_cols] = scaler.transform(df[num_cols])

    return df


def predict(input_data: dict):
    processed = preprocess_input(input_data)
    prediction = model.predict(processed)
    return prediction[0]