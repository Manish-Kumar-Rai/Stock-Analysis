import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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

# Forecasting
st.header('Forecasting')
forecast = reliance_forecasting.forecast(data)
st.write(forecast)

# Plot forecast
st.header('Forecast Plot')
fig, ax = plt.subplots()
reliance_forecasting.plot_forecast(data, forecast, ax)
st.pyplot(fig)

# Summary
st.header('Summary')
summary = reliance_forecasting.summarize_forecast(forecast)
st.write(summary)