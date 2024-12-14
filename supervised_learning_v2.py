# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv("USA_Housing_Dataset.csv")

# Explore the dataset to check column names (uncomment to view column names)
# print(df.head())

# Define features (X) and target (y)
# Replace 'Feature1', 'Feature2', etc., with the actual column names in your dataset.
# Replace 'Target' with the actual name of the column representing the house prices.
X = df[["bedrooms", "bathrooms", "yr_built","sqft_basement", "yr_renovated"]]
y = df["price"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

# Display predictions alongside actual values
results = pd.DataFrame({"Actual": y_test, "Predicted": predictions})
print(results.head())
