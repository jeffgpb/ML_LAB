# %%
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# %%
# function to laod a dataset    
def load_data(data_path):
    return pd.read_csv(data_path)

# %%
data_path = '/home/jeffs/mach_learn/USA_Housing_Dataset.csv'
df_1 = load_data(data_path)


# %%
df_1.head()

# %%
df_1.columns

# %%
#function to split the data into training and testing sets
def split_data(data):
    X = data[['bedrooms', 'bathrooms','condition','yr_built', 'yr_renovated']]
    y = data['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# %%
X_train, X_test, y_train, y_test = split_data(df_1)

# %%
print(X_train)

# %%
print(X_test)

# %%
# print(y_train)
# print y_train with two decimal places and commas for thousands
print(y_train.apply(lambda x: '{:,.2f}'.format(x)))
                

# %%
# print y_test with two decimal places and commas for thousands
print(y_test.apply(lambda x: '{:,.2f}'.format(x)))

# %%
# train a linear regression model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# %%
model = train_model(X_train, y_train)

# %%
print(model)

# %%
# make predictions using the trained model
def predict(model, X_test):
    return model.predict(X_test)

# %%

predictions = predict(model, X_test)

# %%
# print predictions with two decimal places and commas for thousands
print(pd.Series(predictions).apply(lambda x: '{:,.2f}'.format(x)))

# %%
# convert the ouput to a dataframe
def predictions_to_df(predictions):
    return pd.DataFrame(predictions)

# %%
df_predictions = predictions_to_df(predictions)

df_predictions.head()
# convert the number to a string with two decimal places and commas for thousands
df_predictions = df_predictions[0].map(lambda x: '{:,.2f}'.format(x))
print(df_predictions)



# %%
import numpy as np

def round_predictions(predictions):
    return np.round(predictions, 2)

# Example usage
pred_round = round_predictions(predictions)
# print(pred_round)

# calculate the mean squared error
def calculate_mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

mse = calculate_mse(y_test, predictions)
print(mse)

# %%
# convert the output to a dollar value with two decimal places
def convert_to_dollars(predictions):
    return predictions.round(2)

# %%
predictions_dollars = convert_to_dollars(df_predictions)

# %%
print(predictions_dollars)

# %%
# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

# %%
def convert_to_dollars(amount):
    return "${:,.2f}".format(amount)

# MSE value
mse_value = 83297328002.39993

# Convert MSE to a dollar format
mse_dollars = convert_to_dollars(mse_value)

print(f"Mean Squared Error in dollars: {mse_dollars}")

# %%
# function to calculate the root mean squared error
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# %%
#calculate the root mean squared error
rmse = calculate_rmse(y_test, predictions)
print("Root Mean Squared Error:", rmse)

# %%
# compare the predicted values with the actual values
def compare_predictions(predictions, y_test):
    return pd.DataFrame({'Actual': y_test, 'Predicted': predictions})

# %%
table = compare_predictions(predictions_dollars[0], y_test)

# %%
print(table)

# %%
# calculate  the difference between the actual and predicted values in dollars
def calculate_diff(predictions, y_test):
    return predictions - y_test

diff = calculate_diff(predictions, y_test)
print(diff)

# %%
# calculate the percentage difference between the actual and predicted values
def calculate_percentage_diff(predictions, y_test):
    return ((predictions - y_test) / y_test) * 100
diff_percentage = calculate_percentage_diff(predictions, y_test)
print(diff_percentage)

# %%
# print data type of the predictions and y_test
# print(type(predictions))    
# print(type(y_test))
# convert the predictions to a dataframe
def predictions_to_df(predictions):
    return pd.DataFrame(predictions)
predictions = predictions_to_df(predictions)
print(type(predictions))



