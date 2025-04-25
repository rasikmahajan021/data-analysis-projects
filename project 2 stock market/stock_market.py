import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Define Scope and Objective
# Analyzing historical stock prices for Apple, Microsoft, Netflix, and Google


# -------------------------------------------------------------------------------------------------------


# Step 2: Data Collection (Using a Provided Dataset)
 
file_path = "stocks.csv"  # Assume dataset is stored locally
data = pd.read_csv(file_path)
data["Date"] = pd.to_datetime(data["Date"], format="%d-%m-%Y")
print(data.info())
# defining the path and loading the data for analysing.



# -------------------------------------------------------------------------------------------------------



# Step 3: Data Preparation

data.dropna(inplace=True)
print(data.describe())
# This removes any missing values (NaN) from the dataset which can cause error while analysing.



# -------------------------------------------------------------------------------------------------------



# Step 4: Exploratory Data Analysis (EDA)
# used seaborn.lineplot() to visualize stock price trends for each company over time.
# hue="Ticker" used for differentiate the stocks by colour.

plt.figure(figsize=(12,6))
sns.lineplot(data=data, x="Date", y="Close", hue="Ticker")
plt.title("Stock Price Trends Over Time")
plt.show()
# Helps in understanding stock price movement.
# Identifies patterns or anomalies in stock prices.



# -------------------------------------------------------------------------------------------------------



# Step 5: Feature Engineering (Moving Averages)
# Calculates 50-day and 200-day moving averages.
data["MA_50"] = data.groupby("Ticker")["Close"].transform(lambda x: x.rolling(window=50).mean())
data["MA_200"] = data.groupby("Ticker")["Close"].transform(lambda x: x.rolling(window=200).mean())
# The rolling(window=50).mean() function smoothens price fluctuations by averaging the last 50 days.
# A 200-day MA crossing above/below the 50-day MA is often used to predict stock movement.


plt.figure(figsize=(12,6))
for ticker in data["Ticker"].unique():
    subset = data[data["Ticker"] == ticker]
    plt.plot(subset["Date"], subset["Close"], label=f"{ticker} Close", alpha=0.6)
    plt.plot(subset["Date"], subset["MA_50"], label=f"{ticker} MA 50")
    plt.plot(subset["Date"], subset["MA_200"], label=f"{ticker} MA 200")
plt.legend()
plt.title("Stock Prices with Moving Averages")
plt.show()
# Plots the moving averages along with the actual closing price.
# Moving averages help identify trends by filtering out short-term fluctuations.



# -------------------------------------------------------------------------------------------------------



# Step 6: Model Selection (Linear Regression for Stock Price Prediction)

# Converts Date into a numerical feature (day of the year).
data["Day"] = data["Date"].dt.dayofyear
X = data[["Day"]]
y = data["Close"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Splits data into training (80%) and testing (20%) sets.

# Uses Linear Regression to predict stock prices based on Day.
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# Linear regression finds the best-fit line for predicting stock prices.
# Helps understand whether stock prices follow a linear trend over time.



# -------------------------------------------------------------------------------------------------------



# Step 7: Model Evaluation

# Uses mean_squared_error() to measure prediction accuracy.
# Helps determine if Linear Regression is a good fit for stock price prediction.
# Lower MSE means the model is more accurate.
 

plt.figure(figsize=(8,5))
plt.scatter(X_test, y_test, label="Actual Prices", alpha=0.6)
plt.scatter(X_test, y_pred, label="Predicted Prices", alpha=0.6)
plt.legend()
plt.title("Stock Price Prediction (Linear Regression)")
plt.show()

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))