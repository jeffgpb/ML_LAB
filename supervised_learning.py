# Import necessary libraries
"""
This script performs the following tasks:
1. Loads a dataset from a CSV file.
2. Splits the dataset into features and target variable.
3. Splits the data into training and testing sets.
4. Trains a linear regression model on the training data.
5. Makes predictions on the testing data.
6. Prints the first few rows of the dataset.
7. Prints the predictions made by the model.
8. Prints a table comparing actual and predicted values along with the prediction error percentage.
9. Evaluates the model using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).
10. Calculates and prints the Mean Absolute Percentage Error (MAPE).
11. Calculates and prints the Mean Percentage Error (MPE).
12. Writes the results to a CSV file.
13. Reads the results from the CSV file without the index column and loads it into another variable.

Functions:
- load_data(file_name): Loads a dataset from a CSV file.
- split_data(data): Splits the dataset into features and target variable.

Variables:
- filename: The path to the CSV file containing the dataset.
- data: The loaded dataset.
- df: The dataset as a pandas DataFrame.
- X: The features of the dataset.
- y: The target variable of the dataset.
- X_train: The training set features.
- X_test: The testing set features.
- y_train: The training set target variable.
- y_test: The testing set target variable.
- model: The linear regression model.
- predictions: The predictions made by the model on the testing set.
- results: A DataFrame containing the actual values, predicted values, and prediction error percentage.
- mse: The Mean Squared Error of the model.
- rmse: The Root Mean Squared Error of the model.
- mape: The Mean Absolute Percentage Error of the model.
- mpe: The Mean Percentage Error of the model.
- results2: The DataFrame containing the results read from the CSV file.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import math
#function to load dataset from a csv file   
def load_data(file_name):
    data = pd.read_csv(file_name)
    return data

#function to split the dataset into features and target
def split_data(data):
    X = data.drop('price', axis=1)
    y = data['price']
    return X, y

filename = '/home/jeffs/mach_learn/USA_Housing_Dataset.csv'

data =load_data(filename)

df = pd.DataFrame(data)

print(df.head())

# Split data into features (X) and target (y)
X = df[[ 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot'
      ]]
y = df["price"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
print(predictions)

# print a table comparing actual and predicted values wiht column showing prediction error percentage
results = pd.DataFrame({"Actual": y_test, "Predicted": predictions, "Error": np.abs(y_test - predictions) / y_test * 100})
print(results.head(5))
print(results.columns)
# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

rmse = math.sqrt(mse)

print("Root Mean Squared Error:", rmse)

# calculate the mean absolute percentage error
mape = np.mean(np.abs(results['Error']))
print("Mean Absolute Percentage Error:", mape)

# calculate the mean percentage error
mpe = np.mean(results['Error'])

# write the results to a csv file
results.to_csv('results.csv', index=False)

# read the results from the csv file withou the index column and load in other variable
results2 = pd.read_csv('results.csv')