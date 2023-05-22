import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf  # module for scraping data from yahoo finance
from yahoofinancials import YahooFinancials
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential


# starting and ending date of the stocks
start = datetime(2013, 1, 1)
end = datetime(2022, 12, 31)

st.title("Stock Trend Prediction")

user_input = st.text_input("Enter Stock Ticker : ", "SBIN.NS")
if user_input == "":
    e = ValueError("Please enter the stock ticker...")
    st.error(e)
df = yf.download(user_input, start=start, end=end, progress=False)

# Describing Data
st.subheader("Data from 2013-2023")
st.write(df.describe())

# visualizations
st.subheader("Closing Price vs Time chart")
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

ma100 = df.Close.rolling(100).mean()
st.subheader("Closing Price vs Time chart With 100MA")
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader("Closing Price vs Time chart With 100MA & 200MA")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, "r")
plt.plot(ma200, "g")
plt.plot(df.Close, "b")
st.pyplot(fig)

# separating training data and testing data
data_training = pd.DataFrame(df["Close"][:int(len(df)*0.7)])  # for training data taking 70% data
data_testing = pd.DataFrame(df["Close"][int(len(df)*0.7):])  # for testing data taking 30% data

# scaling data between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)
data_testing_array = scaler.fit_transform(data_testing)

# load model 
model = load_model("keras_model.h5")

# testing part 

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
sc = scaler.scale_
# print(scaler, type(scaler))
scale_factor = 1/sc[0]
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor


# final graph 
st.subheader("Predictions vs Original")
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, "b", label="Original Price")
plt.plot(y_predicted, "r", label="Predicted Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig2)
