# Predictive Modeling for Real Estate Prices

The real estate market is one of the most dynamic and complex sectors of the economy. Predicting property prices accurately is a critical task for various stakeholders, including real estate agents, investors, and homebuyers. The ability to estimate property prices based on various factors such as location, size, quality, and other attributes can help make informed decisions and maximize profits.

In this project, the goal is to build a predictive model that estimates the prices of properties based on several input features. These features include the overall quality of the house, the size, the number of rooms, the location, and other relevant attributes. We aim to evaluate and compare several machine learning models to determine the best approach for predicting property prices.

To achieve this, we employ multiple regression techniques, including Linear Regression, Decision Tree, and Random Forest, to understand their performance in predicting housing prices. We will evaluate these models based on key performance metrics such as Root Mean Squared Error (RMSE), R², and Mean Absolute Percentage Error (MAPE) to determine the most effective model for this problem.

By carefully analyzing the results, we aim to identify the best-performing model and provide insights on how to improve predictions for real estate pricing.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Technologies Used](#technologies-used)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Training & Evaluation](#model-training--evaluation)
5. [Model Comparison](#model-comparison)
6. [Results](#results)


## Project Overview

The project aims to predict housing prices based on various features of properties such as the number of bedrooms, size, location, and more. The primary goal is to use machine learning algorithms to make accurate predictions on real estate prices. The following models have been implemented:
- **Linear Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**
- **Ridge Regression**

## Technologies Used

- Python
- Libraries:
  - **pandas**: Data manipulation and analysis.
  - **numpy**: Numerical operations.
  - **matplotlib** and **seaborn**: Visualization.
  - **sklearn**: For implementing machine learning algorithms, preprocessing, and evaluation metrics.
  - **scipy**: Statistical tests.
  
## Data Preprocessing

The dataset is preprocessed before being used to train the models:
1. **Feature Selection**: Only relevant numeric and categorical features are selected.
2. **Scaling**: Standard scaling is applied to numeric features to standardize the data.
3. **Handling Missing Values**: Missing values are handled appropriately.
4. **Categorical Data Encoding**: Categorical variables are converted into numeric features.
  
## Model Training & Evaluation

### 1. **Linear Regression**

The Linear Regression model is trained and evaluated on the scaled features using the following steps:
- Training the model on the training dataset.
- Making predictions on both the training and test sets.
- Evaluating the model using R², RMSE and MAPE.

```python
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred_test = model.predict(X_test_scaled)
```

### **2. Decision Tree Regressor**

The Decision Tree model is trained with a specified maximum depth and evaluated:

- The importance of each feature is visualized.
- The training and test RMSE, R² and MAPE values are calculated.

```python
from sklearn.tree import DecisionTreeRegressor

tree_model = DecisionTreeRegressor(max_depth=5, random_state=42)
tree_model.fit(X_train_scaled, y_train)
y_pred_train = tree_model.predict(X_train_scaled)
```

### **3. Random Forest Regressor**

The Random Forest model is trained with 400 trees, a maximum depth of 10, and other hyperparameters:

- Evaluation metrics (RMSE, R² and MAPE) are calculated for both training and testing datasets.

```python
from sklearn.ensemble import RandomForestRegressor

model_rf = RandomForestRegressor(n_estimators=400, max_depth=10, random_state=42)
model_rf.fit(X_train_scaled, y_train)
```

### **4. Ridge Regression with Cross-Validation**

The Ridge Regression model is trained using cross-validation to determine the best alpha (regularization strength) value. The model is evaluated using R², RMSE and MAPE.


### **Model Comparison**

After training the models, their performances are compared using the following metrics:

- R²: The proportion of the variance in the dependent variable that is predictable from the independent variables.
- RMSE: The Root Mean Squared Error, a measure of the differences between predicted and actual values.
- MAPE (Mean Absolute Percentage Error): Measures the average percentage difference between the predicted and actual values.

## **Results**

After training and evaluating the models, the following results are printed:

- R², RMSE, and MAPE scores for training and testing datasets.
- Feature importance from Decision Tree and Random Forest models.
- Cross-validation scores for Random Forest and Ridge models.

## **Conclusion**

### **Analysis and Selection of the Best Model**
Based on the results of the RMSE, R², and MAPE metrics, we observe that the Random Forest model strikes a good balance between training and testing data. The Random Forest model has an R² of 0.92 on the training set and 0.88 on the test set, indicating that the model captures the data variation well while maintaining good generalization. Additionally, the MAPE for the test is 12.06%, which is relatively low, suggesting that the predictions are, on average, quite accurate.

On the other hand, the Linear Regression model shows a very high R² on the training set (0.94), but the significant difference between training and testing results (0.88 on the test) indicates that the model may be overfitting, which leads to higher errors on the test set. Additionally, the MAPE on the test is higher (12.59%) compared to Random Forest.

Although the Decision Tree model performs reasonably well, with an R² of 0.81 on the test set, it exhibits the highest error (RMSE) and the highest MAPE on the test data, indicating that this model struggles more to generalize.

### **Final Conclusion**:
Among the models evaluated, Random Forest presented the best overall performance, with a good balance between error on the training and test sets. Therefore, the RandomForestRegressor model is the most suitable for predicting property prices, combining accuracy and robustness in generalizing the data.
  
