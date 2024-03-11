import pandas as pd

# Load DataSet
df = pd.read_csv("./dataset_antenna.csv")
# Data preparation

# Data separation as X and y
y = df["s11(dB)"]
X = df.drop("s11(dB)", axis=1)


# Data splitting
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=100
)

## Linear Regression
from sklearn.linear_model import LinearRegression

# Training the model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Applying the model to make a prediction
y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)
# Evaluate model performance
from sklearn.metrics import mean_squared_error, r2_score


def lrperformance():
    lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
    lr_train_r2 = r2_score(y_train, y_lr_train_pred)
    lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
    lr_test_r2 = r2_score(y_test, y_lr_test_pred)
    return {
        "LR MSE (Train): ": lr_train_mse,
        "LR R2 (Train): ": lr_train_r2,
        "LR MSE (Test): ": lr_test_mse,
        "LR R2 (Test): ": lr_test_r2,
    }


# Random Forest
from sklearn.ensemble import RandomForestRegressor

# Training the model
rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(X_train, y_train)

# Applying the model to make a prediction
y_rf_train_pred = rf.predict(X_train)
y_rf_test_pred = rf.predict(X_test)


# Evaluate model performance
def rfperformance():
    rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
    rf_train_r2 = r2_score(y_train, y_rf_train_pred)
    rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
    rf_test_r2 = r2_score(y_test, y_rf_test_pred)
    return {
        "RF MSE (Train): ": rf_train_mse,
        "RF R2 (Train): ": rf_train_r2,
        "RF MSE (Test): ": rf_test_mse,
        "RF R2 (Test): ": rf_test_r2,
    }


# ElasticNet
from sklearn.linear_model import ElasticNet

# Train the model
en = ElasticNet(random_state=100)
en.fit(X_train, y_train)

# Apply the model to make a prediction
y_en_train_pred = en.predict(X_train)
y_en_test_pred = en.predict(X_test)


# Evaluate model performance
def enperformance():
    en_train_mse = mean_squared_error(y_train, y_en_train_pred)
    en_train_r2 = r2_score(y_train, y_en_train_pred)
    en_test_mse = mean_squared_error(y_test, y_en_test_pred)
    en_test_r2 = r2_score(y_test, y_en_test_pred)
    return {
        "EN MSE (Train): ": en_train_mse,
        "EN R2 (Train): ": en_train_r2,
        "EN MSE (Test): ": en_test_mse,
        "EN R2 (Test): ": en_test_r2,
    }


# Lasso
from sklearn.linear_model import Lasso

# Train the model
lasso = Lasso(random_state=100)
lasso.fit(X_train, y_train)
# Apply the model to make a prediction
y_lasso_train_pred = lasso.predict(X_train)
y_lasso_test_pred = lasso.predict(X_test)


# Evaluate model performance
def lassoperformance():
    lasso_train_mse = mean_squared_error(y_train, y_lasso_train_pred)
    lasso_train_r2 = r2_score(y_train, y_lasso_train_pred)
    lasso_test_mse = mean_squared_error(y_test, y_lasso_test_pred)
    lasso_test_r2 = r2_score(y_test, y_lasso_test_pred)
    return {
        "LASSO MSE (Train): ": lasso_train_mse,
        "LASSO R2 (Train): ": lasso_train_r2,
        "LASSO MSE (Test): ": lasso_test_mse,
        "LASSO R2 (Test): ": lasso_test_r2,
    }


# DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor

# Train the model
dt = DecisionTreeRegressor(max_depth=3, random_state=100)
dt.fit(X_train, y_train)

# Apply the model to make a prediction
y_dt_train_pred = dt.predict(X_train)
y_dt_test_pred = dt.predict(X_test)


# Evaluate model performance
def dtperformance():
    dt_train_mse = mean_squared_error(y_train, y_dt_train_pred)
    dt_train_r2 = r2_score(y_train, y_dt_train_pred)
    dt_test_mse = mean_squared_error(y_test, y_dt_test_pred)
    dt_test_r2 = r2_score(y_test, y_dt_test_pred)
    return {
        "DT MSE (Train): ": dt_train_mse,
        "DT R2 (Train): ": dt_train_r2,
        "DT MSE (Test): ": dt_test_mse,
        "DT R2 (Test): ": dt_test_r2,
    }


# Predicting Values With Different Algo

# New data point
# new_data = {
#     "Freq(GHz)": 1.380,
#     "length of patch in mm": 31.11000,
#     "width of patch in mm": 40.40000,
#     "Slot length in mm": 13.94,
#     "slot width in mm": 2.86,
# }


def predictingValues(new_data):
    # print(type(data))
    # new_data =

    # Convert new data to a dataframe
    new_df = pd.DataFrame([new_data])

    # Predict using Linear Regression
    lr_predicted_value = lr.predict(new_df)[0]

    # Predict using Random Forest
    rf_predicted_value = rf.predict(new_df)[0]

    # Predict using Lasso
    lasso_predicted_value = lasso.predict(new_df)[0]

    # Predict using ElasticNet
    en_predicted_value = en.predict(new_df)[0]

    # Predict using DecisionTreeRegressor
    dt_predicted_value = dt.predict(new_df)[0]

    return {
        "Linear": lr_predicted_value,
        "Random Forest ": rf_predicted_value,
        "ElasticNet": en_predicted_value,
        "Lasso": lasso_predicted_value,
        "Decision Tree": dt_predicted_value,
    }


# Example usage:
# S11 = predictingValues(new_data)
# print("S11 parameter:", S11)

import math


def calculate_S11(new_data):
    # Constants
    c = 3e8  # Speed of light in m/s
    epsilon_r = 4.4  # Dielectric constant of the substrate
    h = 1.6e-3  # Thickness of the substrate in meters

    # Extract data from new_data dictionary
    freq_GHz = new_data["Freq(GHz)"]
    length_patch_mm = new_data["length of patch in mm"]
    width_patch_mm = new_data["width of patch in mm"]

    # Convert dimensions to meters
    length_patch_m = length_patch_mm / 1000
    width_patch_m = width_patch_mm / 1000

    # Calculate effective dielectric constant
    epsilon_eff = (epsilon_r + 1) / 2 + ((epsilon_r - 1) / 2) * (
        1 + 12 * h / length_patch_m
    ) ** (-0.5)

    # Calculate resonant frequency
    f0 = c / (
        2 * math.pi * math.sqrt(epsilon_eff) * math.sqrt(length_patch_m * width_patch_m)
    )

    # Calculate S11 parameter
    S11 = (f0 - freq_GHz * 1e9) / f0

    return S11


# Example usage:
# S11 = calculate_S11(new_data)
# print("S11 parameter:", S11)
