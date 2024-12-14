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

# Load the dataset
data_path = '/home/jeffs/mach_learn/USA_Housing_Dataset.csv'
df_1 = load_data(data_path)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = split_data(df_1)

# Train the model
model = train_model(X_train, y_train)

# Make predictions
predictions = predict(model, X_test)
predictions_df = predictions_to_df(predictions)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

# Print evaluation metrics
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

# Compare predictions with actual values
comparison_table = compare_predictions(predictions_df['Predicted'], y_test)

# Calculate differences using numeric predictions
diff = calculate_diff(predictions_df['Predicted'], y_test)
diff_percentage = calculate_percentage_diff(predictions_df['Predicted'], y_test)

# Add differences to the comparison table
comparison_table['Difference'] = diff
comparison_table['Percentage Difference'] = diff_percentage

# Format the predictions for display (AFTER numeric operations)
comparison_table['Predicted'] = format_predictions(comparison_table['Predicted'])

# Print the formatted table
print(comparison_table.head())

# Print differences (optional)
print("Differences:")
print(diff.head())

print("Percentage Differences:")
print(diff_percentage.head())

# read the csv file results.csv and convert it to a DataFrame
results = pd.read_csv('results.csv')
print(results)
