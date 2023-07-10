import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
# Dummy data for demonstration
X = np.random.random((100, 10))  # Input features
y = np.random.random((100, 1))  # Target values
# Define the model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=10))  # First hidden layer
model.add(Dense(64, activation='relu'))  # Second hidden layer
model.add(Dense(1))  # Output layer
# Define the number of folds
k = 5

# Perform K-Fold cross-validation
kf = KFold(n_splits=k)
mse_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Compile and train the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

    # Evaluate the model on test data
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)

# Print the mean MSE score across folds
mean_mse = np.mean(mse_scores)
print("Mean MSE Score:", mean_mse)
