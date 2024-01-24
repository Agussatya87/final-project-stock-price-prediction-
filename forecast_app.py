import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import yfinance as yf
from datetime import date
import pickle

st.set_page_config(layout='wide', initial_sidebar_state='expanded')

st.sidebar.header('Dashboard')

st.title('BRI Stock Forecast Web')

stock = 'BBRI.JK'  # Removed unnecessary parentheses

n_months = 1
period = 30 * n_months

@st.cache_data
def load_data(ticker):
    START = "2014-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data = load_data(stock)

# Load the model using pickle
with open("prophet_model.pkl", "rb") as model_file:
    model = pd.read_pickle(model_file)

# Load forecast data
forecast = pd.read_csv("forecast_data.csv")

# Display raw data
st.subheader('Raw data')
st.write(data.tail(30))

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

def plot_raw_data_ma100(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'].rolling(100).mean(), name="MA 100"))
    fig.update_layout(title_text='Time Series data vs Moving Average 100', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data_ma100(data)

def plot_raw_data_ma(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'].rolling(100).mean(), name="MA 100"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'].rolling(200).mean(), name="MA 200"))
    fig.update_layout(title_text='Time Series data vs Moving Average 100 vs Moving Average 200', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data_ma(data)

# Rename columns in forecast DataFrame
forecast = forecast.rename(columns={'ds': 'date', 'yhat': 'close'})

# Display forecast data
st.subheader(f'Forecast data for the next {n_months} months')
st.write(forecast[['date', 'close']].tail(n_months * period))

st.write(f'Forecast plot for {n_months} months')
fig1 = plot_plotly(model, forecast) 
st.plotly_chart(fig1)
