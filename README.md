Here is the complete README file for the project **Rain Prediction System**, combining all relevant details:

---

# Rain Prediction System

## Overview
The **Rain Prediction System** is a machine learning-based project designed to predict the likelihood of rain using weather data. It leverages real-time and historical weather data, processes it, and uses an XGBoost classifier to make predictions. The system also includes features like data visualization and interactive options for users.

This project was developed by **Aadil Rahman**, **Judin Jomon**, and **A Likhit**.

---

## Features
- **Real-Time Weather Data**: Fetches live weather data using the OpenWeatherMap API.
- **Historical Data Preloading**: Loads historical weather data for model training.
- **Rain Prediction**: Predicts rain likelihood using an XGBoost classifier.
- **Data Visualization**: Displays weather trends and distributions using Plotly.
- **Interactive Console**: Allows users to capture new data, train models, and make predictions.

---

## Installation

### Prerequisites
- Python 3.8 or later
- Required Python libraries:
  - `numpy`
  - `pandas`
  - `requests`
  - `scikit-learn`
  - `xgboost`
  - `plotly`

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rain-prediction-system.git
   cd rain-prediction-system
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up the OpenWeatherMap API key:
   - Replace the placeholder in the code (`openweathermap_api_key`) with your API key.

---

## Usage

### Running the System
Execute the main script:
```bash
python main.py
```

### Options in the Console
1. **Capture and Save New Data**: Fetches real-time weather data and saves it to a CSV file.
2. **Train Model**: Trains an XGBoost model using available weather data.
3. **Predict Rain**: Predicts whether it will rain based on the latest weather data.
4. **Exit**: Exits the program.

---

## Project Structure

```
rain-prediction-system/
├── main.py                # Entry point of the application
├── requirements.txt       # Dependencies for the project
├── weather_data.csv       # CSV file storing captured weather data (auto-generated)
├── README.md              # Project documentation
└── other scripts/modules  # Supporting functions and utilities
```

---

## Documentation

### Data Capturing and Processing
- Real-time weather data is fetched from OpenWeatherMap API.
- Sensor data (temperature and humidity) is simulated for testing purposes.
- Data is synchronized into a structured format for storage and analysis.

### Model Training
- Features used: temperature, humidity, cloud cover, wind speed, wind direction.
- Target variable: binary classification (rain/no rain).
- Model pipeline includes scaling and training with XGBoost.

### Visualization
- Line plots for temperature, humidity, and cloud cover over time.
- Pie charts for weather condition distribution.

---

## Contributing

We welcome contributions! To contribute:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature description"
   ```
4. Push to your branch:
   ```bash
   git push origin feature-name
   ```
5. Submit a pull request.

---

## License

This project is licensed under the MIT License.

---

## Authors

Developed by:
- **Aadil Rahman**
- **Judin Jomon**
- **A Likhit**

Date: August 18, 2024

--- 

