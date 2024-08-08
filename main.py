import sys
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

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
reliance_forecasting.plot_acf(data['Differenced_Close'].dropna(), ax=ax_acf)
st.pyplot(fig_acf)

fig_pacf, ax_pacf = plt.subplots()
reliance_forecasting.plot_pacf(data['Differenced_Close'].dropna(), ax=ax_pacf)
st.pyplot(fig_pacf)

# Forecasting models
selected_model = st.sidebar.selectbox("Select Model", ["ARIMA", "SARIMA", "Exponential Smoothing"])

st.header(f'Forecasting with {selected_model}')
if selected_model == "ARIMA":
    arima_forecast_diff, arima_mse_diff = reliance_forecasting.fit_arima(data)
    st.write(f'ARIMA MSE (Differenced): {arima_mse_diff}')
    st.write(arima_forecast_diff)
elif selected_model == "SARIMA":
    sarima_forecast_diff, sarima_mse_diff = reliance_forecasting.fit_sarima(data)
    st.write(f'SARIMA MSE (Differenced): {sarima_mse_diff}')
    st.write(sarima_forecast_diff)
else:
    exp_smoothing_forecast_diff, exp_smoothing_mse_diff = reliance_forecasting.fit_exponential_smoothing(data)
    st.write(f'Exponential Smoothing MSE (Differenced): {exp_smoothing_mse_diff}')
    st.write(exp_smoothing_forecast_diff)

# Plot forecast
st.header('Forecast Plot Comparison')
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data['Date'][-100:], data['Differenced_Close'][-100:], label='Actual')
if selected_model == "ARIMA":
    ax.plot(data['Date'][-30:], arima_forecast_diff, label='ARIMA Forecast')
elif selected_model == "SARIMA":
    ax.plot(data['Date'][-30:], sarima_forecast_diff, label='SARIMA Forecast')
else:
    ax.plot(data['Date'][-30:], exp_smoothing_forecast_diff, label='Exponential Smoothing Forecast')
ax.set_xlabel('Date')
ax.set_ylabel('Differenced Close Price')
ax.set_title('Reliance Industries - Forecast Comparison')
ax.legend()
st.pyplot(fig)

# Future date prediction
st.sidebar.subheader("Future Date Prediction")
future_date_str = st.sidebar.text_input("Enter Future Date (YYYY-MM-DD)", (data['Date'].iloc[-1] + pd.Timedelta(days=30)).strftime('%Y-%m-%d'))
try:
    future_date = datetime.strptime(future_date_str, '%Y-%m-%d')
    if future_date <= data['Date'].iloc[-1]:
        st.sidebar.error("Future date must be after the last date in the dataset.")
    else:
        predicted_price = reliance_forecasting.predict_price_on_date(data, selected_model, future_date)
        st.header('Predict Future Price')
        st.write(f"Predicted Price on {future_date.strftime('%Y-%m-%d')}: {predicted_price:.2f}")
except ValueError:
    st.sidebar.error("Please enter a valid date in the format YYYY-MM-DD.")

# Summary
st.header('Model Performance Summary')
st.write(f'ARIMA MSE (Differenced): {arima_mse_diff}')
st.write(f'SARIMA MSE (Differenced): {sarima_mse_diff}')
st.write(f'Exponential Smoothing MSE (Differenced): {exp_smoothing_mse_diff}')
