
## **Documentation for House Price Prediction Code**

### **Overview**
This Python script implements a pipeline to predict house prices using linear regression. It processes a dataset, trains a regression model, and evaluates its predictions by calculating key metrics and generating a detailed comparison table. The results are saved as a CSV file.

---

### **Libraries Used**
1. **pandas**: For data manipulation and analysis.
2. **numpy**: For numerical computations.
3. **sklearn.model_selection.train_test_split**: To split data into training and testing sets.
4. **sklearn.linear_model.LinearRegression**: To train a linear regression model.
5. **sklearn.metrics.mean_squared_error**: To evaluate model performance.

---

### **Functions**

#### **1. `load_data(data_path)`**
- **Description**: Loads a dataset from a CSV file.
- **Parameters**:
  - `data_path` (str): Path to the CSV file.
- **Returns**: A pandas DataFrame containing the dataset.

---

#### **2. `split_data(data)`**
- **Description**: Splits the dataset into training and testing sets for features and the target variable.
- **Parameters**:
  - `data` (DataFrame): Input dataset containing features and target.
- **Returns**:
  - `X_train`, `X_test`: Training and testing feature sets.
  - `y_train`, `y_test`: Training and testing target values.

---

#### **3. `train_model(X_train, y_train)`**
- **Description**: Trains a linear regression model on the training data.
- **Parameters**:
  - `X_train` (DataFrame): Training feature set.
  - `y_train` (Series): Training target values.
- **Returns**: Trained LinearRegression model.

---

#### **4. `predict(model, X_test)`**
- **Description**: Generates predictions using the trained model on the test set.
- **Parameters**:
  - `model`: Trained LinearRegression model.
  - `X_test` (DataFrame): Testing feature set.
- **Returns**: A numpy array of predicted values.

---

#### **5. `predictions_to_df(predictions)`**
- **Description**: Converts predictions into a pandas DataFrame.
- **Parameters**:
  - `predictions` (array): Predicted values.
- **Returns**: DataFrame with a `Predicted` column.

---

#### **6. `compare_predictions(predictions, y_test)`**
- **Description**: Compares predicted values with actual target values.
- **Parameters**:
  - `predictions` (array): Predicted values.
  - `y_test` (Series): Actual target values.
- **Returns**: DataFrame with `Actual` and `Predicted` columns.

---

#### **7. `calculate_diff(predictions, y_test)`**
- **Description**: Calculates the difference between predicted and actual values.
- **Parameters**:
  - `predictions` (array): Predicted values.
  - `y_test` (Series): Actual target values.
- **Returns**: A Series containing the differences.

---

#### **8. `calculate_percentage_diff(predictions, y_test)`**
- **Description**: Calculates the percentage difference between predicted and actual values.
- **Parameters**:
  - `predictions` (array): Predicted values.
  - `y_test` (Series): Actual target values.
- **Returns**: A Series containing the percentage differences.

---

#### **9. `format_predictions(predictions)`**
- **Description**: Formats numeric values as strings with a dollar sign and two decimal places (e.g., `$1,234.56`).
- **Parameters**:
  - `predictions` (Series): Numeric values to be formatted.
- **Returns**: A Series of formatted strings.

---

#### **10. `percentage_change(new_value, old_value)`**
- **Description**: Computes the percentage change from `old_value` to `new_value`.
- **Parameters**:
  - `new_value` (float): The new value.
  - `old_value` (float): The baseline value.
- **Returns**: Percentage change as a float.

---

### **Main Workflow**

1. **Load the Dataset**:
   - The dataset is loaded from the path specified by `data_path`.

2. **Split the Data**:
   - Features (`X`) and target (`y`) are extracted, and the dataset is split into training and testing sets.

3. **Train the Model**:
   - A linear regression model is trained using the training data (`X_train`, `y_train`).

4. **Make Predictions**:
   - The model predicts house prices for the test set (`X_test`).

5. **Evaluate the Model**:
   - Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) are calculated to measure model performance.

6. **Compare Predictions with Actual Values**:
   - A comparison table is created, including actual values, predicted values, differences, and percentage differences.

7. **Format and Save Results**:
   - Columns (`Actual`, `Predicted`, `Difference`) are formatted in dollar notation.
   - The final comparison table is saved to a CSV file named `results.csv`.

---

### **Key Outputs**
1. **Evaluation Metrics**:
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)

2. **Comparison Table**:
   - `Actual`: Actual house prices (formatted in dollars).
   - `Predicted`: Predicted house prices (formatted in dollars).
   - `Difference`: Absolute differences between actual and predicted values (formatted in dollars).
   - `Percentage Difference`: Relative differences in percentages.

3. **CSV Output**:
   - A detailed comparison table is written to `results.csv`.

---

### **Example Usage**

```bash
$ python house_price_prediction.py
Mean Squared Error: 83297328002.40
Root Mean Squared Error: 288612.76
```

Example of the output table:
```
       Actual    Predicted   Difference   Percentage Difference
0  $600,000.00  $574,269.53  $25,730.47              4.29%
1  $370,000.00  $288,406.78  $81,593.22             22.05%
...
```

---

### **Limitations**
1. **Feature Selection**:
   - Only a subset of features (`bedrooms`, `bathrooms`, `condition`, `yr_built`, `yr_renovated`) is used, which may not fully capture price determinants.
   
2. **Evaluation Metrics**:
   - MSE and RMSE are included, but additional metrics (e.g., \( R^2 \) or MAE) might provide more insights.

3. **Formatting Assumptions**:
   - Dollar formatting may not be suitable for non-financial use cases.
 