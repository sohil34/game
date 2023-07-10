import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from sklearn.model_selection import train_test_split
# Load the stock price data
data = pd.read_csv("tesla-stock-price.csv")

# Extract the stock prices
prices = data["Close"].values

# Normalize the prices
prices = (prices - np.mean(prices)) / np.std(prices)

# Prepare the data for training the RNN
X = []
y = []

# Generate input sequences and corresponding target values
sequence_length = 10
for i in range(len(prices) - sequence_length):
    X.append(prices[i:i+sequence_length])
    y.append(prices[i+sequence_length])

X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Define the RNN model
model = Sequential()
model.add(SimpleRNN(32, input_shape=(sequence_length, 1)))
model.add(Dense(1))

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=16)
loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)