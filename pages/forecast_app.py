import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import yfinance as yf
from datetime import date
import numpy as np

st.set_page_config(layout='wide', initial_sidebar_state='expanded')

st.title('BRI Stock Prices Predict')

stock = 'BBRI.JK'

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

st.subheader('Raw data')
st.write(data.tail(30))

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
df_train = df_train.rename(columns={"Date": "ds", "Close": "prices"})

model = Prophet()
model.fit(df_train)

future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)

forecast['date']  = forecast['ds']
forecast['close'] = forecast['yhat']

st.subheader(f'Forecast data for the next {n_months} months')
st.write(forecast[['date', 'close']].tail(n_months * period))

df_test = data.tail(n_months * period) 
df_test = df_test.rename(columns={"Date": "ds", "Close": "prices"}) 

df_test = pd.merge(df_test, forecast[['ds', 'yhat']], how='left', left_on='ds', right_on='ds')
df_test = df_test.rename(columns={"yhat": "yhat_forecast"})

# from sklearn.metrics import mean_absolute_error, mean_squared_error
# mae = mean_absolute_error(df_test['y'], df_test['yhat_forecast'])
# mse = mean_squared_error(df_test['y'], df_test['yhat_forecast'])
# rmse = np.sqrt(mse)

# df_test['mape'] = (abs(df_test['y'] - df_test['yhat_forecast']) / abs(df_test['y'])) * 100
# df_test['mspe'] = ((df_test['y'] - df_test['yhat_forecast'])**2 / df_test['y']**2) * 100
# df_test['rmspe'] = np.sqrt(df_test['mspe'])

st.write(f'Forecast plot for {n_months} months')
fig1 = plot_plotly(model, forecast) 
st.plotly_chart(fig1)

# st.subheader('Percentage Errors:')
# st.write(f'Mean Absolute Percentage Error (MAPE): {df_test["mape"].mean():.2f}%')
# st.write(f'Mean Squared Percentage Error (MSPE): {df_test["mspe"].mean():.2f}%')
# st.write(f'Root Mean Squared Percentage Error (RMSPE): {df_test["rmspe"].mean():.2f}%')
