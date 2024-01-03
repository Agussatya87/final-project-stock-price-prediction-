import streamlit as st
from datetime import date
import pandas as pd
import yfinance as yf
import numpy as np
from keras.models import load_model
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = load_model('stock_predict_model.h5')

# Function to load data
@st.cache
def load_data(ticker, start, end):
    data = yf.download(ticker, start, end)
    data.reset_index(inplace=True)
    return data

# Function to create sequences for prediction
def create_sequences_for_prediction(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        seq = data[i:i+sequence_length, 0]
        sequences.append(seq)
    return np.array(sequences)

# Streamlit app
st.title('Stock Prediction App')

# Load and preprocess data
START = "2014-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
selected_stock = st.selectbox('Select dataset for prediction', ['AAPL', 'GOOG', 'MSFT'])
data = load_data(selected_stock, START, TODAY)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Create sequences for prediction
sequence_length = 50
input_data = create_sequences_for_prediction(scaled_data, sequence_length)

# Make predictions
predicted_prices = model.predict(input_data)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Display raw data
st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
st.subheader('Raw data plot')
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Actual Close Price"))
fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
st.plotly_chart(fig)

# Display predicted prices
st.subheader('Predicted Prices')
st.write(pd.DataFrame({'Date': data['Date'].iloc[sequence_length-1:], 'Predicted Close': predicted_prices.flatten()}))

# Plot predicted prices
st.subheader('Predicted Prices Plot')
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'].iloc[sequence_length-1:], y=predicted_prices.flatten(), name="Predicted Close Price"))
fig.layout.update(title_text='Predicted Prices vs Actual Close Price', xaxis_rangeslider_visible=True)
st.plotly_chart(fig)
