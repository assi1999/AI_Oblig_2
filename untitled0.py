# Importing the relevant libraries
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Collect data from the CSV file
data = pd.read_csv('/TSLA-1.csv')

# Preprocess Data
data['Date'] = pd.to_datetime(data['Date'])  # Convert the 'Date' column to a datetime format
data.set_index('Date', inplace=True)  # Set 'Date' as the index in the dataset

# Split Data
train_data, test_data = train_test_split(data, test_size=0.2)  # Split data for training and testing sets

# Features of the CSV file
features = ['Open', 'Low', 'High', 'Volume']  # Define the features used for prediction, using 'Open', 'Low', 'High', 'Volume'

# Train the model with Linear Regression
model = LinearRegression()  # Create a Linear Regression model
model.fit(train_data[features], train_data['Close'])  # Train the model on the training data, using features to predict 'Close'

# Make the predictions
predictions = model.predict(test_data[features])  # The model makes predictions on the test data

# Evaluate the model to assess its performance
mse = mean_squared_error(test_data['Close'], predictions)  # Calculate mean squared error
rmse = np.sqrt(mse)  # Calculate the root mean squared error

print(f"Root Mean Squared Error: {rmse}")  # Print the root mean squared error

# Calculate the percentage score based on RMSE
percentage_score = 100 - (rmse / test_data['Close'].mean()) * 100
print(f"Prediction Percentage Score: {percentage_score:.2f}%")

# Function to find the nearest available date
def find_nearest_date(date, data):
    idx = (data.index - date).argmin()
    return data.index[idx]

# Predict the stock price for a specific date
get_date_from_user = input("Enter the desired date (Format: YYYY-MM-DD): ")
specific_date = pd.to_datetime(get_date_from_user)  # Convert user input to the correct Timestamp
nearest_date = find_nearest_date(specific_date, data)
specific_data = data.loc[nearest_date, features].values.reshape(1, -1)
specific_prediction = model.predict(specific_data)[0]
print(f"Predicted Price for {get_date_from_user}: {specific_prediction}")


