import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings("ignore")

# Function to load data
def load_data():
    ticker_symbol = 'RELIANCE.BO'
    reliance_stock = yf.Ticker(ticker_symbol)
    historical_data = reliance_stock.history(period='10y')
    historical_data.reset_index(inplace=True)
    historical_data['Date'] = pd.to_datetime(historical_data['Date'])
    return historical_data

# Function to plot data
def plot_data(data, ax):
    ax.plot(data['Date'], data['Close'], label='Close Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.set_title('Reliance Industries - Historical Close Prices')
    ax.legend()

# Make data stationary
def make_stationary(data):
    data['Differenced_Close'] = data['Close'].diff()
    return data

# Custom Plot ACF and PACF
def plot_acf_custom(data, ax):
    plot_acf(data, ax=ax, lags=40)

def plot_pacf_custom(data, ax):
    plot_pacf(data, ax=ax, lags=40)

# ARIMA model
def fit_arima(data):
    arima_model = ARIMA(data['Differenced_Close'].dropna(), order=(5, 0, 0))
    arima_fit = arima_model.fit()
    arima_forecast_diff = arima_fit.forecast(steps=30)
    arima_mse_diff = mean_squared_error(data['Differenced_Close'][-30:], arima_forecast_diff)
    return arima_forecast_diff, arima_mse_diff

# SARIMA model
def fit_sarima(data):
    sarima_model = SARIMAX(data['Differenced_Close'].dropna(), order=(1, 0, 1), seasonal_order=(1, 0, 1, 12))
    sarima_fit = sarima_model.fit(disp=False)
    sarima_forecast_diff = sarima_fit.forecast(steps=30)
    sarima_mse_diff = mean_squared_error(data['Differenced_Close'][-30:], sarima_forecast_diff)
    return sarima_forecast_diff, sarima_mse_diff

# Exponential Smoothing model
def fit_exponential_smoothing(data):
    exp_smoothing_model = ExponentialSmoothing(data['Differenced_Close'].dropna(), trend='add', seasonal='add', seasonal_periods=12)
    exp_smoothing_fit = exp_smoothing_model.fit()
    exp_smoothing_forecast_diff = exp_smoothing_fit.forecast(steps=30)
    exp_smoothing_mse_diff = mean_squared_error(data['Differenced_Close'][-30:], exp_smoothing_forecast_diff)
    return exp_smoothing_forecast_diff, exp_smoothing_mse_diff

# Function to predict future price
def predict_future_price(data, future_date):
    model = ARIMA(data['Close'], order=(5, 0, 0))
    model_fit = model.fit()
    
    # Ensure future_date is a datetime object and get only the date part
    if isinstance(future_date, pd.Timestamp):
        future_date = future_date.date()
    
    # Determine the number of days between the last known date and the future date
    last_known_date = data['Date'].iloc[-1].date()  # Convert to date object
    days_ahead = (future_date - last_known_date).days
    
    # Check if the future date is in the past
    if days_ahead <= 0:
        raise ValueError("The future date must be after the last known date in the dataset.")
    
    # Forecast future price
    forecast = model_fit.forecast(steps=days_ahead)
    future_price = forecast.iloc[-1] if len(forecast) > 0 else np.nan
    return future_price
