# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample dataset
data = {
    "Size (sq. ft)": [1500, 2000, 2500, 3000, 3500],
    "Bedrooms": [3, 4, 3, 5, 4],
    "Age (years)": [10, 15, 20, 5, 7],
    "Price": [300000, 400000, 350000, 500000, 450000]
}
df = pd.DataFrame(data)

# Split data into features (X) and target (y)
X = df[["Size (sq. ft)", "Bedrooms", "Age (years)"]]
y = df["Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
print(X_train)
print(y_train)
print(X_test)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
print(predictions)
# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)


import math

mse = 8321005.917168107
rmse = math.sqrt(mse)
print("Root Mean Squared Error:", rmse)
# create a table comparing actual and predicted values
results = pd.DataFrame({"Actual": y_test, "Predicted": predictions})

print(results.head(5))