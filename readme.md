# Weather Classification System (End-to-End ML Project)

## Overview

This project implements an end-to-end Machine Learning pipeline to classify weather conditions into four categories:

* Rainy
* Sunny
* Cloudy
* Snowy

The system is built using multiple classification algorithms, optimized through hyperparameter tuning, and deployed using a FastAPI backend with a Streamlit-based interactive UI.

---

## Problem Statement

### Business Problem

To automate the classification of weather conditions based on environmental parameters such as temperature, humidity, wind speed, and atmospheric conditions. This helps in improving decision-making for domains like agriculture, logistics, and forecasting.

### Machine Learning Problem

A **supervised multi-class classification** problem where the goal is to predict the target variable **Weather Type** using numerical and categorical features.

---

## Dataset Description

The dataset is synthetically generated and includes the following features:

### Numerical Features:

* Temperature
* Humidity
* Wind Speed
* Precipitation (%)
* Atmospheric Pressure
* UV Index
* Visibility (km)

### Categorical Features:

* Cloud Cover
* Season
* Location

### Target Variable:

* Weather Type (Rainy, Sunny, Cloudy, Snowy)

---

## Key Observations

* Dataset is balanced across all classes
* No missing values or duplicate records
* Presence of outliers in features like Temperature, Humidity, and Pressure
* Strong relationships observed between features and target variable

---

## ML Pipeline

### 1. Data Preprocessing

* Outlier handling using IQR-based capping
* One-Hot Encoding for categorical variables
* Feature Scaling using StandardScaler
* Train-Test Split (80-20)

---

### 2. Model Building

The following classification models were implemented:

* Logistic Regression
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)

---

### 3. Model Evaluation

Models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score

Since the dataset is balanced, **Accuracy and F1-score** were used as primary evaluation metrics.

---

### 4. Hyperparameter Tuning

* GridSearchCV (Exhaustive search)
* RandomizedSearchCV (Efficient search)

Tuning was performed on:

* Random Forest
* SVM

---

### 5. Final Model Selection

Random Forest was selected as the final model based on:

* Strong performance (F1-score ~0.91)
* Robustness to outliers
* Better generalization
* Faster inference compared to SVM

---

## Inference Pipeline

A production-ready inference pipeline was built to ensure consistency between training and prediction phases:

**Steps:**

1. Input data ingestion
2. Outlier handling (using saved bounds)
3. One-hot encoding
4. Feature alignment
5. Scaling
6. Model prediction

All preprocessing artifacts (scaler, columns, outlier bounds) are persisted using `joblib`.

---

## Project Structure

```
weather-ml-classification/
│
├── app/
│   ├── main.py
│   ├── schema.py
│   ├── utils.py
│
├── models/
│   ├── rf_weather_model.pkl
│   ├── scaler.pkl
│   ├── columns.pkl
│   ├── outlier_bounds.pkl
│
├── data/
│   ├── weather_classification_data.csv
│
├── ui/
│   ├── streamlit_app.py
│
├── requirements.txt
└── README.md
```

---

## API (FastAPI)

### Endpoint:

```
POST /predict
```

### Input:

JSON payload with weather parameters

### Output:

Predicted weather type

---

## User Interface (Streamlit)

The Streamlit app provides:

* User input form for predictions
* API integration for real-time inference
* Interactive visualizations:

  * Temperature distribution
  * Humidity vs Temperature
  * Precipitation trends

---

## How to Run the Project

### Step 1: Install Dependencies

```
pip install -r requirements.txt
```

### Step 2: Start FastAPI Server

```
uvicorn app.main:app --reload
```

### Step 3: Run Streamlit App

```
streamlit run ui/streamlit_app.py
```

---

## Results

* Best Model: Random Forest
* Accuracy: ~91%
* F1-Score: ~0.91

---

## Key Learnings

* Importance of proper data preprocessing
* Handling outliers in synthetic datasets
* Model comparison and evaluation
* Hyperparameter tuning strategies
* Building production-ready ML pipelines
* Integrating ML models with APIs and UI

---
