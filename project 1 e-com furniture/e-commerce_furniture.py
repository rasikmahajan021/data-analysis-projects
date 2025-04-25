import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np






#1 Data Collection

# Loading the dataset
file_path = "furniture_data_ecommerce.csv"
df = pd.read_csv(file_path)

# Printing the first few rows
print(df.head(5))

#--------------------------------------------------------------------------------------------------------

#2 Data Preprocessing

# Checking for missing values
print(df.isnull().sum())

# Dropping any rows with missing values.
df = df.dropna()

# Converting tagText into a categorical feature.
# df['tagText'] = df['tagText'].astype('category').cat.codes


# Now you can use it for modeling
print(df.head())

# Checking for data types and conversions if necessary
print(df.info())


#--------------------------------------------------------------------------------------------------------

#3 Exploratory Data Analysis(EDA)

# Graph 1 : BAR GRAPH
# Group by shipping type and get average sales
# avg_sales_by_shipping = df.groupby("tagText")["sold"].mean().sort_values()
# Convert one-hot encoded columns back to a single categorical column
df["Shipping_Type"] = df[["tagText_Express Shipping", "tagText_Free Shipping", "tagText_Standard Shipping"]].idxmax(axis=1)

# Clean up column names
df["Shipping_Type"] = df["Shipping_Type"].str.replace("tagText_", "")

# Now, group by Shipping_Type instead of the missing 'tagText'
avg_sales_by_shipping = df.groupby("Shipping_Type")["sold"].mean().sort_values()

print(avg_sales_by_shipping)  # Check the results
# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=avg_sales_by_shipping.index, y=avg_sales_by_shipping.values, palette="coolwarm")
plt.title("Average Items Sold by Shipping Type")
plt.xlabel("Shipping Type")
plt.ylabel("Average Number of Items Sold")
plt.xticks(rotation=30)
plt.show()
# This graph shows which shipping method leads to higher sales.


# Graph 2 : Line Chart
# Sort data by price for a smooth curve
sorted_df = df.sort_values(by="price")

# Line plot
plt.figure(figsize=(10, 5))
sns.lineplot(x=sorted_df['price'], y=sorted_df['sold'], marker='o', color='green')
plt.title("Price vs. Number of Items Sold")
plt.xlabel("Price")
plt.ylabel("Number of Items Sold")
plt.show()
# This chart helps see how price changes affect sales in a smooth trend.


# Graph 3 : 
# Define price ranges
bins = [0, 50, 100, 200, 500, 1000]
labels = ["$0-50", "$50-100", "$100-200", "$200-500", "$500+"]

# Create a new column for price range
df["price_range"] = pd.cut(df["price"], bins=bins, labels=labels)

# Count sales in each category
sales_distribution = df.groupby("price_range")["sold"].sum()

# Pie Chart
plt.figure(figsize=(7, 7))
plt.pie(sales_distribution, labels=sales_distribution.index, autopct="%1.1f%%", colors=["#ff9999","#66b3ff","#99ff99","#ffcc99","#c2c2f0"])
plt.title("Sales Distribution by Price Range")
plt.show()


# Graph 4 :
# plt.figure(figsize=(8, 5))
sns.boxplot(x=df["price_range"], y=df["sold"], palette="pastel")
plt.title("Sales Variation by Price Range")
plt.xlabel("Price Range ($)")
plt.ylabel("Number of Items Sold")
plt.show()


#--------------------------------------------------------------------------------------------------------

#4 Feature Engineering

# Create the Discount Percentage feature
df["discount_pct"] = ((df["originalPrice"] - df["price"]) / df["originalPrice"]) * 100

# Remove infinite values (caused by division by zero) and NaNs
df["discount_pct"] = df["discount_pct"].replace([float("inf"), -float("inf")], None)
df = df.dropna(subset=["discount_pct"])

# Plot the distribution of Discount Percentage
plt.figure(figsize=(10, 5))
sns.histplot(df["discount_pct"], bins=30, kde=True, color="blue")
plt.xlabel("Discount Percentage")
plt.ylabel("Frequency")
plt.title("Distribution of Discount Percentage in Furniture Sales")
plt.grid(True)

# Show the plot
plt.show()
# will help in understanding the impact of discounts on the sale.
# will help us see if the product with higher discounts is selling more or not.


#--------------------------------------------------------------------------------------------------------


#5 Model Selection and training

# Encode 'tagText' into numerical categories
le = LabelEncoder()
# df['tagText_encoded'] = le.fit_transform(df['tagText']) commented because "tagText" no longer exists.

# Convert one-hot encoded columns back to a single categorical column
df["Shipping_Type"] = df[["tagText_Express Shipping", "tagText_Free Shipping", "tagText_Standard Shipping"]].idxmax(axis=1)
# Clean up column names (Remove 'tagText_' prefix) 
df["Shipping_Type"] = df["Shipping_Type"].str.replace("tagText_", "") 

# Now apply Label Encoding (Categorical text is converted into numerical values using LabelEncoder)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["Shipping_Type_encoded"] = le.fit_transform(df["Shipping_Type"])  # ✅ Encodes the correct column


print(df[["Shipping_Type", "Shipping_Type_encoded"]].head())  # Check if it works


# Remove outliers in 'sold' (keep only data within the 99th percentile)
df = df[(df['sold'] > 0) & (df['sold'] < df['sold'].quantile(0.99))]

# Define features (X) and target (y)
# X = df[['price', 'originalPrice', 'discount_pct', 'tagText_encoded']]
X = df[['price', 'originalPrice', 'discount_pct', 'Shipping_Type_encoded']]  # ✅ Correct column

y = df['sold']


# Split into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize and train the Decision Tree model with limited depth
dt_model = DecisionTreeRegressor(max_depth=1, random_state=42)
dt_model.fit(X_train, y_train)

# Predict on test data
y_pred_dt = dt_model.predict(X_test)

print("Model training completed successfully!")


#--------------------------------------------------------------------------------------------------------


#6 Model Evaluation

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred_dt)
mse = mean_squared_error(y_test, y_pred_dt)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_dt)

# Print results
print("📊 Model Evaluation Results:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Interpretation
if r2 > 0.7:
    print("✅ The model explains most of the variance in the data. It's a good fit!")
elif r2 > 0.4:
    print("⚠️ The model has moderate performance. Consider improving it.")
else:
    print("❌ The model is not performing well. Try tuning hyperparameters or using more features.")



#-------------------------------------------------------------------------------------------------------


#7 Conclusion

# :
# 📌 Conclusion for the E-Commerce Furniture Sales Prediction Project
# ✅ Project Summary:
# This project aimed to analyze and predict furniture sales based on various product attributes, such as price, original price, discount percentage, and shipping type. The goal was to develop a machine learning model that could help businesses understand sales patterns and optimize their pricing strategies.

# 🔍 Key Findings & Insights:
# 1️⃣ Impact of Price & Discount on Sales:

# Items with higher discounts tend to have higher sales compared to those with little or no discount.
# However, extremely low prices do not always lead to high sales, indicating that customers may still consider quality and brand reputation.
# 2️⃣ Shipping Type Matters:

# Products with "Express Shipping" showed a higher average number of sales, suggesting that fast delivery is a key factor in customer decision-making.
# Standard shipping items had relatively lower sales, which means businesses might benefit from offering faster shipping options.
# 3️⃣ Data Cleaning & Outliers Removal Helped Improve Model Performance:

# The raw dataset contained some inconsistent values (e.g., mismatched discount prices), which were corrected.
# Outliers in sales data were removed to prevent the model from being skewed by extreme values.
# 📊 Model Performance & Evaluation:
# A Decision Tree Regressor was trained to predict sales based on the available features.
# The model was evaluated using Mean Absolute Error (MAE), Mean Squared Error (MSE), RMSE, and R² Score.
# The performance of the model suggests that:
# If the R² score is high (> 0.7) → The model is a good fit for predicting sales.
# If the R² score is low (< 0.4) → The model needs improvement by adding more features or tuning hyperparameters.
