import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime 
import yfinance as yf  # module for scraping data from yahoo finance
from yahoofinancials import YahooFinancials
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

# starting and ending date of the stocks 
start = datetime(2013, 1, 1)
end = datetime(2022, 12, 31)

# taking past 10 years data 
df = yf.download('SBIN.NS', start=start, end=end, progress=False)
df.reset_index(inplace=True)

# deleting columns - date and Adj Close, which are not required
df.drop(['Date'], axis=1, inplace=True)
df.drop(['Adj Close'], axis=1, inplace=True)
# print(df)

# plotting data for 100 jumps, taking average value per 100 days
ma100 = df.Close.rolling(100).mean()
# print(ma100)
plt.figure(figsize=(12, 6))
plt.plot(df.Close)
# plotting data for 200 jumps, taking average value per 200 days
ma200 = df.Close.rolling(200).mean()
plt.plot(ma100, "red", label="100 MA")
plt.plot(ma200, "green", label="200 MA")
plt.legend()
plt.show()

# separating training data and testing data
data_training = pd.DataFrame(df["Close"][:int(len(df)*0.7)])  # for training data taking 70% data
data_testing = pd.DataFrame(df["Close"][int(len(df)*0.7):])   # for testing data taking 30% data

print(data_training.shape)
print(data_testing.shape)

# scaling data between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)
data_testing_array = scaler.fit_transform(data_testing)

x_train = []
y_train = []

# storing values for 100 days interval
for i in range(100, data_training.shape[0]):
    x_train.append(data_training_array[i-100: i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train) 

# ML model
model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.3))

model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1))
model.summary()

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(x_train, y_train, epochs=50)
model.save("keras_model.h5")

past_100_days = data_testing.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)
print(input_data.shape)
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])
    
x_test, y_test = np.array(x_test), np.array(y_test)

# making predictions 

y_predicted = model.predict(x_test)
sc = scaler.scale_
# print(scaler.scale_)
scale_factor = 1/sc[0]
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor
plt.figure(figsize=(12, 6))
plt.plot(y_test, "b", label="Original Price")
plt.plot(y_predicted, "r", label="Predicted Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()
