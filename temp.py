# Importing the relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Collect data from the CSV file
data = pd.read_csv('/TSLA-1.csv')

# Preprocess Data

data.set_index('Date', inplace=True)  # Set 'Date' as the index in the dataset

# Split Data
train_data, test_data = train_test_split(data, test_size=0.2)  # Split data for training and testing sets

# Features of the CSV file
features = ['Open', 'Low', 'High', 'Volume']  # Define the features used for prediction, using 'Open', 'Low', 'High', 'Volume'

# Train the model with Regression
model = LinearRegression()  # Create a Linear Regression model
model.fit(train_data[features], train_data['Close'])  # Train the model on the training data, using features to predict 'Close'

# Make the predictions
predictions = model.predict(test_data[features])  # The model makes predictions on the test data

# Visualize results
fig, ax = plt.subplots(figsize=(20, 10))
plt.plot(test_data.index, test_data['Close'], label='Actual Price', color='blue')  # Plot actual 'Close' prices
plt.plot(test_data.index, predictions, label='Predicted Price', color='red')  # Plot predicted 'Close' prices
plt.legend()
plt.title('Tesla Inc Stock Price Prediction')
plt.xlabel('Date')  # Label for the x-axis
plt.ylabel('Price')  # Label for the y-axis
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Ensure labels are not cut off
plt.show()


"""
Why did we use Linear Regression?
Well in this code we used linear reggresion on the tranning data using features ['Open', 'Low', 'High', 'Volume'] to predict the 'Close' price of the Tesla Inc stock.
Then it stores the predictions on the data set in the variable 'predictions'.
We chose linear reggresion, becuase of simplicity and the possible linear connection between the variables. 
This might be to simple for some datasets, but worked in this instance.
There are other factor that also play a major role in determining the outcome of the stock price, but in this simple methode, we didin't regard those other factors.
This methode of machine learning is also applicable for continues price predictions.
In addition, it makes the feature variable consisting of ['Open', 'Low', 'High', 'Volume'] more easily accessable for the traning set, to further predict a algorithm for price determning.  
"""