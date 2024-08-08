import sys
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

# Ensure the current directory is in the Python path
sys.path.append(os.path.dirname(__file__))

# Load the converted Python script
import reliance_forecasting

# Title
st.title('Reliance Forecasting')

# Load data
st.header('Load Data')
data = reliance_forecasting.load_data()
st.write(data.head())

# Data visualization
st.header('Data Visualization')
fig, ax = plt.subplots()
reliance_forecasting.plot_data(data, ax)
st.pyplot(fig)

# Make data stationary
st.header('Make Data Stationary')
data = reliance_forecasting.make_stationary(data)
st.write(data[['Date', 'Differenced_Close']].head())

# ACF and PACF plots
st.header('ACF and PACF Plots')
fig_acf, ax_acf = plt.subplots()
reliance_forecasting.plot_acf_custom(data['Differenced_Close'].dropna(), ax=ax_acf)
st.pyplot(fig_acf)

fig_pacf, ax_pacf = plt.subplots()
reliance_forecasting.plot_pacf_custom(data['Differenced_Close'].dropna(), ax=ax_pacf)
st.pyplot(fig_pacf)

# Forecasting with models
st.header('Forecasting with ARIMA')
arima_forecast_diff, arima_mse_diff = reliance_forecasting.fit_arima(data)
st.write(f'ARIMA MSE (Differenced): {arima_mse_diff}')
st.write(arima_forecast_diff)

st.header('Forecasting with SARIMA')
sarima_forecast_diff, sarima_mse_diff = reliance_forecasting.fit_sarima(data)
st.write(f'SARIMA MSE (Differenced): {sarima_mse_diff}')
st.write(sarima_forecast_diff)

st.header('Forecasting with Exponential Smoothing')
exp_smoothing_forecast_diff, exp_smoothing_mse_diff = reliance_forecasting.fit_exponential_smoothing(data)
st.write(f'Exponential Smoothing MSE (Differenced): {exp_smoothing_mse_diff}')
st.write(exp_smoothing_forecast_diff)

# Plot forecast comparison
st.header('Forecast Plot Comparison')
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data['Date'][-100:], data['Differenced_Close'][-100:], label='Actual')
ax.plot(data['Date'][-30:], arima_forecast_diff, label='ARIMA Forecast')
ax.plot(data['Date'][-30:], sarima_forecast_diff, label='SARIMA Forecast')
ax.plot(data['Date'][-30:], exp_smoothing_forecast_diff, label='Exponential Smoothing Forecast')
ax.set_xlabel('Date')
ax.set_ylabel('Differenced Close Price')
ax.set_title('Reliance Industries - Forecast Comparison')
ax.legend()
st.pyplot(fig)

# Add a section for future date prediction
st.header('Predict Future Price')
last_date_in_data = data['Date'].iloc[-1].date()  # Get the last date in the dataset

# Allow the user to select a future date for prediction
future_date = st.date_input('Select a future date for prediction', min_value=last_date_in_data + timedelta(days=1))

# Predict the future price based on the selected date
if future_date:
    future_price = reliance_forecasting.predict_future_price(data, future_date)
    st.write(f'Predicted price on {future_date}: {future_price}')

# Summary
st.header('Model Performance Summary')
st.write(f'ARIMA MSE (Differenced): {arima_mse_diff}')
st.write(f'SARIMA MSE (Differenced): {sarima_mse_diff}')
st.write(f'Exponential Smoothing MSE (Differenced): {exp_smoothing_mse_diff}')
