# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Function to load a dataset
def load_data(data_path):
    return pd.read_csv(data_path)

# Function to split the data into training and testing sets
def split_data(data):
    X = data[['bedrooms', 'bathrooms', 'condition', 'yr_built', 'yr_renovated']]
    y = data['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Function to train a linear regression model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Function to make predictions using the trained model
def predict(model, X_test):
    return model.predict(X_test)

# Convert predictions to a DataFrame
def predictions_to_df(predictions):
    return pd.DataFrame(predictions, columns=['Predicted'])

# Function to compare predictions with actual values
def compare_predictions(predictions, y_test):
    # Ensure both predictions and y_test are numeric
    predictions = predictions.squeeze()  # Flatten DataFrame if necessary
    return pd.DataFrame({'Actual': y_test.values, 'Predicted': predictions})

# Function to calculate the difference between actual and predicted values
def calculate_diff(predictions, y_test):
    return predictions - y_test

# Function to calculate the percentage difference between actual and predicted values
def calculate_percentage_diff(predictions, y_test):
    return ((predictions - y_test) / y_test) * 100

# Function to format predictions for display
def format_predictions(predictions):
    return predictions.apply(lambda x: "${:,.2f}".format(x))

# Function to calculate percentage difference between two numbers
def percentage_change(new_value, old_value):
    if old_value == 0:
        return float('inf')  # Avoid division by zero
    return ((new_value - old_value) / old_value) * 100

# Example usage of percentage difference calculation
value1 = 4.25
value2 = 3.10
percentage_diff = percentage_change(value1, value2)
print(f"Percentage difference between {value1} and {value2}: {percentage_diff:.2f}%")

# Load the dataset
data_path = '/home/jeffs/mach_learn/USA_Housing_Dataset.csv'
df_1 = load_data(data_path)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = split_data(df_1)

# Train the model
model = train_model(X_train, y_train)

# Make predictions
predictions = predict(model, X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

# Print evaluation metrics
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

table = compare_predictions(predictions, y_test)

print(table)


# Calculate differences using numeric predictions
diff = calculate_diff(predictions, y_test)
diff.index = table.index  # Ensure indices match
print(diff)
# add the differences to the comparison table
table['Difference'] = diff
print(table)

diff_percentage = calculate_percentage_diff(predictions, y_test)
print(diff_percentage)
# add the percentage differences to the comparison table
table.index = diff_percentage.index
table['Percentage Difference'] = diff_percentage
print(table)

#  convert Actual, Predicted , Differences columns to dollars format for printing   
table['Actual'] = format_predictions(table['Actual'])
table['Predicted'] = format_predictions(table['Predicted'])
table['Difference'] = format_predictions(table['Difference'])
print(table)

# write the results to a csv file 
table.to_csv('results.csv', index=False)

print("Results written to results.csv")

