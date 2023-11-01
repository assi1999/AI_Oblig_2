# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv('TSLA.csv')

# ordere data by 'Date'
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values('Date', inplace=True)
data.set_index('Date', inplace=True)

# Checking for missing values 
if data.isnull().sum().sum() != 0:
    data.dropna(inplace=True)

# Features and target for the regression model we arw using
features = ['Open']
target = 'Close'

# Split data sequentially (80% train, 20% test)
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

# Train a linear regression model
model = LinearRegression()
model.fit(train_data[features], train_data[target])

# Predict on test data
predictions = model.predict(test_data[features])

# Scatter plot of Open vs. Close prices with regression line
plt.figure(figsize=(20, 10))
plt.scatter(test_data['Open'], test_data['Close'], color='blue', label='Actual Price')
plt.plot(test_data['Open'], predictions, color='red', label='Regression Line')
plt.legend()
plt.title('Tesla Inc Stock Price Prediction')
plt.xlabel('Open Price')
plt.ylabel('Close Price')
plt.tight_layout()
plt.show()

#Evaluate model using RMSE
mse = mean_squared_error(test_data[target], predictions)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

# Calculate and display percentage score
percentage_score = 100 - (rmse / test_data[target].mean()) * 100
print(f"Prediction Percentage Score: {percentage_score:.2f}%")

# Function to find the nearest available date
def find_nearest_date(date, data):
    return data.index[np.abs(data.index - date).argmin()]

# Predict the stock price for a specific date
try:
    get_date_from_user = input("Enter the desired date (Format: YYYY-MM-DD): ")
    specific_date = pd.to_datetime(get_date_from_user)
    nearest_date = find_nearest_date(specific_date, data)
    specific_data = data.loc[nearest_date, features].values.reshape(1, -1)
    specific_prediction = model.predict(specific_data)[0]
    print(f"Predicted Price for {specific_date.strftime('%Y-%m-%d')}: {specific_prediction:.2f}")
except:
    print("Invalid date format. Please enter the date in the format YYYY-MM-DD.")


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
