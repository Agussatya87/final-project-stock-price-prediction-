import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import yfinance as yf
from datetime import date
import pickle

st.set_page_config(layout='wide', initial_sidebar_state='expanded')

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

df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

model = Prophet()
model.fit(df_train)

future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)

forecast['date']  = forecast['ds']
forecast['close'] = forecast['yhat']

# Display forecast data
st.subheader(f'Forecast data for the next {n_months} months')
st.write(forecast[['date', 'close']].tail(n_months * period))

st.write(f'Forecast plot for {n_months} months')
fig1 = plot_plotly(model, forecast) 
st.plotly_chart(fig1)
