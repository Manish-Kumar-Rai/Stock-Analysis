import sys
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure the current directory is in the Python path
sys.path.append(os.path.dirname(__file__))

# Load the converted Python script
import reliance_forecasting

# Set Streamlit page configuration
st.set_page_config(
    page_title="Reliance Forecasting",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set style for plots
sns.set_style("whitegrid")

# Title
st.title('Reliance Industries Stock Price Forecasting')

# Introduction
st.markdown("""
Welcome to the Reliance Industries Stock Price Forecasting app. This application uses historical stock price data to forecast future stock prices using various time series models, including ARIMA, SARIMA, and Exponential Smoothing. Explore the data, visualize trends, and compare forecast performance.
""")

# Sidebar
st.sidebar.header("User Options")
selected_model = st.sidebar.selectbox("Select Model for Forecasting", ["ARIMA", "SARIMA", "Exponential Smoothing"])
st.sidebar.write("### Model Parameters")
st.sidebar.slider("Select Forecast Horizon (Days)", 10, 100, 30, step=10)

# Load data
st.header('Load Data')
with st.spinner('Loading data...'):
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
col1, col2 = st.columns(2)

with col1:
    st.subheader('ACF Plot')
    fig_acf, ax_acf = plt.subplots()
    reliance_forecasting.plot_acf(data['Differenced_Close'].dropna(), ax=ax_acf)
    st.pyplot(fig_acf)

with col2:
    st.subheader('PACF Plot')
    fig_pacf, ax_pacf = plt.subplots()
    reliance_forecasting.plot_pacf(data['Differenced_Close'].dropna(), ax=ax_pacf)
    st.pyplot(fig_pacf)

# Forecasting
st.header('Forecasting')

if selected_model == "ARIMA":
    st.subheader('Forecasting with ARIMA')
    forecast_diff, mse_diff = reliance_forecasting.fit_arima(data)
elif selected_model == "SARIMA":
    st.subheader('Forecasting with SARIMA')
    forecast_diff, mse_diff = reliance_forecasting.fit_sarima(data)
else:
    st.subheader('Forecasting with Exponential Smoothing')
    forecast_diff, mse_diff = reliance_forecasting.fit_exponential_smoothing(data)

st.write(f'MSE (Differenced): {mse_diff}')
st.write(forecast_diff)

# Plot forecast
st.header('Forecast Plot Comparison')
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data['Date'][-100:], data['Differenced_Close'][-100:], label='Actual')
ax.plot(data['Date'][-30:], forecast_diff, label=f'{selected_model} Forecast')
ax.set_xlabel('Date')
ax.set_ylabel('Differenced Close Price')
ax.set_title('Reliance Industries - Forecast Comparison')
ax.legend()
st.pyplot(fig)


if selected_model == "ARIMA":
    # Forecasting with ARIMA
    st.header('Forecasting with ARIMA')
    arima_forecast_diff, arima_mse_diff = reliance_forecasting.fit_arima(data)
    st.write(f'ARIMA MSE (Differenced): {arima_mse_diff}')
    st.write(arima_forecast_diff)
elif selected_model == "SARIMA":
    # Forecasting with SARIMA
    st.header('Forecasting with SARIMA')
    sarima_forecast_diff, sarima_mse_diff = reliance_forecasting.fit_sarima(data)
    st.write(f'SARIMA MSE (Differenced): {sarima_mse_diff}')
    st.write(sarima_forecast_diff)
else:
    # Forecasting with Exponential Smoothing
    st.header('Forecasting with Exponential Smoothing')
    exp_smoothing_forecast_diff, exp_smoothing_mse_diff = reliance_forecasting.fit_exponential_smoothing(data)
    st.write(f'Exponential Smoothing MSE (Differenced): {exp_smoothing_mse_diff}')
    st.write(exp_smoothing_forecast_diff)   


# Model Performance Summary
st.header('Model Performance Summary')
if selected_model == "ARIMA":
    st.write(f'ARIMA MSE (Differenced): {arima_mse_diff}')
elif selected_model == "SARIMA":
    st.write(f'SARIMA MSE (Differenced): {sarima_mse_diff}')
else:
    st.write(f'Exponential Smoothing MSE (Differenced): {exp_smoothing_mse_diff}')








st.write(f'Exponential Smoothing MSE (Differenced): {exp_smoothing_mse_diff}')

# Footer
st.markdown("""
---
**Note:** This application is for educational purposes only. The forecasts generated are based on historical data and may not reflect future market conditions.
""")
