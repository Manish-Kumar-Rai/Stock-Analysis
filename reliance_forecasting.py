import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# Function to load data
def load_data():
    ticker_symbol = 'RELIANCE.BO'
    reliance_stock = yf.Ticker(ticker_symbol)
    historical_data = reliance_stock.history(period='10y')
    historical_data.reset_index(inplace=True)
    historical_data['Date'] = pd.to_datetime(historical_data['Date'].dt.strftime('%Y-%m-%d'))
    return historical_data

# Function to plot data
def plot_data(data, ax):
    ax.plot(data['Date'], data['Close'], label='Close Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.set_title('Reliance Industries - Historical Close Prices')
    ax.legend()

# Function to make data stationary and plot ACF/PACF
def make_stationary(data):
    # Differencing the data
    data['Differenced_Close'] = data['Close'].diff().dropna()

    # ADF test
    adf_result_diff = adfuller(data['Differenced_Close'].dropna())
    print(f'ADF Statistic (Differenced): {adf_result_diff[0]}')
    print(f'p-value (Differenced): {adf_result_diff[1]}')

    # KPSS test
    kpss_result_diff, _, _, _ = kpss(data['Differenced_Close'].dropna(), regression='c')
    print(f'KPSS Statistic (Differenced): {kpss_result_diff}')

    # Plot differenced data
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'][1:], data['Differenced_Close'][1:])
    plt.title('Differenced Data')
    plt.xlabel('Date')
    plt.ylabel('Differenced Price')
    plt.show()

    # Plot ACF and PACF
    plot_acf(data['Differenced_Close'].dropna(), lags=40)
    plt.show()

    plot_pacf(data['Differenced_Close'].dropna(), lags=40)
    plt.show()

    return data

# Function to fit ARIMA model
def fit_arima(data,days):
    arima_model_diff = ARIMA(data['Differenced_Close'].dropna(), order=(5, 0, 0))
    arima_fit_diff = arima_model_diff.fit()
    arima_forecast_diff = arima_fit_diff.forecast(steps=days)
    arima_mse_diff = mean_squared_error(data['Differenced_Close'][-30:], arima_forecast_diff)
    print(f'ARIMA MSE (Differenced): {arima_mse_diff}')
    return arima_forecast_diff, arima_mse_diff

# Function to fit SARIMA model
def fit_sarima(data,days):
    sarima_model_diff = SARIMAX(data['Differenced_Close'].dropna(), order=(1, 0, 1), seasonal_order=(1, 0, 1, 12))
    sarima_fit_diff = sarima_model_diff.fit(disp=False)
    sarima_forecast_diff = sarima_fit_diff.forecast(steps=days)
    sarima_mse_diff = mean_squared_error(data['Differenced_Close'][-30:], sarima_forecast_diff)
    print(f'SARIMA MSE (Differenced): {sarima_mse_diff}')
    return sarima_forecast_diff, sarima_mse_diff

# Function to fit Exponential Smoothing model
def fit_exponential_smoothing(data,days):
    exp_smoothing_model_diff = ExponentialSmoothing(data['Differenced_Close'].dropna(), trend='add', seasonal='add', seasonal_periods=12)
    exp_smoothing_fit_diff = exp_smoothing_model_diff.fit()
    exp_smoothing_forecast_diff = exp_smoothing_fit_diff.forecast(steps=days)
    exp_smoothing_mse_diff = mean_squared_error(data['Differenced_Close'][-30:], exp_smoothing_forecast_diff)
    print(f'Exponential Smoothing MSE (Differenced): {exp_smoothing_mse_diff}')
    return exp_smoothing_forecast_diff, exp_smoothing_mse_diff

# Main script to execute functions
if __name__ == "__main__":
    data = load_data()

    # Make data stationary
    data = make_stationary(data)

    # Fit models
    arima_forecast_diff, arima_mse_diff = fit_arima(data)
    sarima_forecast_diff, sarima_mse_diff = fit_sarima(data)
    exp_smoothing_forecast_diff, exp_smoothing_mse_diff = fit_exponential_smoothing(data)

    # Plot forecasts
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'][-100:], data['Differenced_Close'][-100:], label='Actual')
    plt.plot(data['Date'][-30:], arima_forecast_diff, label='ARIMA Forecast')
    plt.plot(data['Date'][-30:], sarima_forecast_diff, label='SARIMA Forecast')
    plt.plot(data['Date'][-30:], exp_smoothing_forecast_diff, label='Exponential Smoothing Forecast')
    plt.legend()
    plt.show()

    # Compare model performance
    print(f'ARIMA MSE (Differenced): {arima_mse_diff}')
    print(f'SARIMA MSE (Differenced): {sarima_mse_diff}')
    print(f'Exponential Smoothing MSE (Differenced): {exp_smoothing_mse_diff}')
