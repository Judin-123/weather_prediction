import numpy as np
import requests
import json
import time
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
import csv
from datetime import datetime, timedelta
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline

# OpenWeatherMap API credentials
openweathermap_api_key = "15e1c3e3523435c267a19c32d770616f"
lat = 12.9716  # Bangalore latitude
lon = 77.5946  # Bangalore longitude

def get_dht_data():
    # Simulate sensor data for testing purposes
    temperature = 25.0  # Example temperature
    humidity = 50.0     # Example humidity
    return temperature, humidity

def get_openweathermap_data():
    base_url = "http://pro.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": openweathermap_api_key,
        "units": "metric"
    }
    response = requests.get(base_url, params=params)
    return response.json()

def sync_data(temperature, humidity, weather_data):
    wind_speed = weather_data['wind']['speed']
    wind_direction = weather_data['wind']['deg']
    weather_condition = weather_data['weather'][0]['main']
    cloud_cover = weather_data['clouds']['all']
    return {
        'temperature': temperature,
        'humidity': humidity,
        'cloud_cover': cloud_cover,
        'wind_speed': wind_speed,
        'wind_direction': wind_direction,
        'weather_condition': weather_condition
    }

def capture_and_process_data():
    temperature, humidity = get_dht_data()
    weather_data = get_openweathermap_data()
    if weather_data is None:
        print("Failed to fetch weather data.")
        return None

    synced_data = sync_data(temperature, humidity, weather_data)
    print("Captured data:", synced_data)
    return synced_data

def save_data_to_csv(data, filename='weather_data.csv'):
    fieldnames = ['timestamp', 'temperature', 'humidity', 'cloud_cover', 'wind_speed', 'wind_direction', 'weather_condition']
    data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

def load_data_from_csv(filename='weather_data.csv'):
    try:
        df = pd.read_csv(filename)
        print(f"Successfully read {len(df)} rows from {filename}")
        return df
    except pd.errors.ParserError as e:
        print(f"Parser error: {e}")
        df = pd.read_csv(filename, on_bad_lines='skip')
        return df
    except FileNotFoundError:
        print(f"Warning: File {filename} not found. Returning empty DataFrame.")
        return pd.DataFrame(columns=['timestamp', 'temperature', 'humidity', 'cloud_cover', 'wind_speed', 'wind_direction', 'weather_condition'])

def prepare_data_for_training(data):
    data['rain'] = data['weather_condition'].apply(lambda x: 1 if 'rain' in str(x).lower() else 0)
    feature_cols = ['temperature', 'humidity', 'cloud_cover', 'wind_speed', 'wind_direction']
    target_col = 'rain'
    return data, feature_cols, target_col

def train_xgb_classifier(data):
    data, feature_cols, target_col = prepare_data_for_training(data)
    X_train, X_test, y_train, y_test = train_test_split(data[feature_cols], data[target_col], test_size=0.2, random_state=42, stratify=data[target_col])

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', xgb.XGBClassifier(
            max_depth=5,
            learning_rate=0.1,
            n_estimators=100,
            reg_alpha=0.1,
            reg_lambda=0.1,
            eval_metric='logloss',
            use_label_encoder=False
        ))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1')
    print(f"Cross-validated F1 scores: {scores}")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return pipeline

def predict_rain(clf, data):
    feature_cols = ['temperature', 'humidity', 'cloud_cover', 'wind_speed', 'wind_direction']
    X = data[feature_cols].values.reshape(1, -1)
    prediction = clf.predict(X)
    return "Rain" if prediction[0] == 1 else "No Rain"

def preload_weather_data(filename='weather_data.csv'):
    if os.path.exists(filename):
        print("Data already preloaded. Skipping.")
        return

def plot_weather_data(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
def main():
