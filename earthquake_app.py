import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import shap
from scipy.stats import poisson

# --------------------------
# Configuration
# --------------------------
st.set_page_config(layout="wide", page_title="AI Earthquake Forecaster")
st.title("🌋 AI-Powered Earthquake Prediction System")


# --------------------------
# Data Loading & Preprocessing
# --------------------------
@st.cache_data
def load_data():
    # Load historical data
    hist_df = pd.read_csv("earthquake_data_1900-2023.csv")
    hist_df['time'] = pd.to_datetime(hist_df['time'])

    # Load tectonic plate data
    plates = gpd.read_file("https://raw.githubusercontent.com/fraxen/tectonicplates/master/GeoJSON/PB2002_plates.json")

    # Load real-time data
    realtime_url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.csv"
    realtime_df = pd.read_csv(realtime_url)

    return hist_df, plates, realtime_df


hist_df, plates_df, realtime_df = load_data()


# --------------------------
# Feature Engineering
# --------------------------
def create_features(df):
    # Time-based features
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['day_of_year'] = df['time'].dt.dayofyear

    # Spatial features
    df['plate_boundary_distance'] = calculate_plate_distance(df, plates_df)

    # Temporal patterns
    df = calculate_seismic_gaps(df)

    # Magnitude bins
    df['magnitude_category'] = pd.cut(df['mag'],
                                      bins=[0, 3, 5, 7, 10],
                                      labels=['micro', 'minor', 'strong', 'major'])
    return df


hist_df = create_features(hist_df)


# --------------------------
# Hybrid Prediction Model
# --------------------------
class EarthquakePredictor:
    def __init__(self):
        self.magnitude_model = self.build_magnitude_model()
        self.temporal_model = self.build_temporal_model()
        self.scaler = MinMaxScaler()

    def build_magnitude_model(self):
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(30, 10)),
            Dropout(0.3),
            LSTM(64),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(loss='mse', optimizer='adam')
        return model

    def build_temporal_model(self):
        model = RandomForestClassifier(n_estimators=200,
                                       class_weight='balanced',
                                       random_state=42)
        return model

    def train_models(self, X_train, y_mag_train, y_time_train):
        # Magnitude prediction
        self.magnitude_model.fit(X_train, y_mag_train,
                                 epochs=50,
                                 batch_size=32,
                                 validation_split=0.2,
                                 verbose=0)

        # Temporal prediction
        self.temporal_model.fit(X_train, y_time_train)

    def predict(self, X):
        mag_pred = self.magnitude_model.predict(X)
        time_prob = self.temporal_model.predict_proba(X)[:, 1]
        return mag_pred, time_prob


# --------------------------
# Streamlit Interface
# --------------------------
def main():
    st.sidebar.header("Model Configuration")

    # Model parameters
    lookback = st.sidebar.slider("Lookback period (days)", 30, 365, 90)
    forecast_window = st.sidebar.slider("Forecast window (days)", 7, 30, 14)
    risk_threshold = st.sidebar.slider("Risk threshold (%)", 10, 50, 25)

    # Initialize predictor
    predictor = EarthquakePredictor()

    # Prepare data
    X, y_mag, y_time = prepare_training_data(hist_df, lookback)
    predictor.train_models(X, y_mag, y_time)

    # --------------------------
    # Real-time Monitoring
    # --------------------------
    st.header("🌍 Real-time Seismic Monitoring")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Recent Earthquakes")
        st.map(realtime_df[['latitude', 'longitude']].rename(columns={
            'latitude': 'lat',
            'longitude': 'lon'
        }))

    with col2:
        st.subheader("Tectonic Stress Map")
        plot_stress_map(hist_df, plates_df)

    # --------------------------
    # Prediction Dashboard
    # --------------------------
    st.header("🔮 AI Forecast")

    # Make predictions
    latest_data = prepare_prediction_data(hist_df, lookback)
    mag_pred, time_prob = predictor.predict(latest_data)

    # Display results
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Magnitude Forecast")
        st.metric("Next Significant Event", f"{mag_pred[0][0]:.1f} M")

        fig1 = plot_magnitude_trend(hist_df, mag_pred)
        st.pyplot(fig1)

    with col4:
        st.subheader("Temporal Probability")
        st.metric(f"Next {forecast_window}-day Risk", f"{time_prob[0] * 100:.1f}%")

        fig2 = plot_probability_heatmap(time_prob, forecast_window)
        st.pyplot(fig2)

    # --------------------------
    # Alert System
    # --------------------------
    if time_prob[0] > (risk_threshold / 100) or mag_pred[0][0] >= 5.0:
        st.error("🚨 HIGH SEISMIC RISK ALERT 🚨")
        st.warning(f"Probability of significant earthquake in next {forecast_window} days: {time_prob[0] * 100:.1f}%")

        if st.button("Generate Detailed Report"):
            generate_risk_report(mag_pred, time_prob, forecast_window)

    # --------------------------
    # Model Interpretation
    # --------------------------
    with st.expander("Model Explainability"):
        st.subheader("SHAP Feature Importance")
        explainer = shap.TreeExplainer(predictor.temporal_model)
        shap_values = explainer.shap_values(X)
        fig3 = shap.summary_plot(shap_values, X, feature_names=hist_df.columns)
        st.pyplot(fig3)


# --------------------------
# Helper Functions
# --------------------------
def prepare_training_data(df, lookback):
    """Convert time series to supervised learning format"""
    # Implementation details omitted for brevity
    return X, y_mag, y_time


def plot_stress_map(quakes, plates):
    """Visualize seismic stress accumulation"""
    # Implementation details omitted for brevity
    return fig


def generate_risk_report(mag_pred, time_prob, window):
    """Create PDF risk assessment"""
    # Implementation details omitted for brevity
    return report


if __name__ == "__main__":
    main()