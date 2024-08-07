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

warnings.filterwarnings("ignore")

# Function to load data
def load_data():
    ticker_symbol = 'RELIANCE.BO'
    reliance_stock = yf.Ticker(ticker_symbol)
    historical_data = reliance_stock.history(period='10y')
    historical_data.reset_index(inplace=True)
    historical_data['Date'] = historical_data['Date'].dt.strftime('%Y-%m-%d')
    historical_data['Date'] = pd.to_datetime(historical_data['Date'])
    return historical_data

# Function to plot data
def plot_data(data, ax):
    ax.plot(data['Date'], data['Close'], label='Close Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.set_title('Reliance Industries - Historical Close Prices')
    ax.legend()

# Function to forecast data
def forecast(data):
    model = ARIMA(data['Close'], order=(5, 1, 0))
    model_fit = model.fit(disp=0)
    forecast_data = model_fit.forecast(steps=30)
    return forecast_data

# Function to plot forecast
def plot_forecast(data, forecast, ax):
    ax.plot(data['Date'], data['Close'], label='Actual')
    forecast_dates = pd.date_range(start=data['Date'].iloc[-1], periods=len(forecast)+1, closed='right')
    ax.plot(forecast_dates, forecast, label='Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.set_title('Reliance Industries - Forecast')
    ax.legend()

# Function to summarize forecast
def summarize_forecast(forecast):
    return forecast.describe()
