# ========================================
# STEP 1: IMPORT LIBRARIES & LOAD DATASET
# ========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = "Electric Vehicle Sales by State in India.csv"
df = pd.read_csv(file_path)

# Display dataset information
print("Dataset Overview:\n")
print(df.info())
print("\nFirst 5 Rows of Data:\n")
print(df.head())




# ========================================
# STEP 2: DATA PREPROCESSING
# ========================================

# Convert 'Year' to integer
df['Year'] = df['Year'].astype(int)

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Check for missing values
print("\nMissing Values in Each Column:\n")
print(df.isnull().sum())

# Fill missing values
df['EV_Sales_Quantity'].fillna(df['EV_Sales_Quantity'].median(), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)





# ========================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ========================================

# ---- EV Sales Trend Over the Years ----
sales_summary = df.groupby("Year")["EV_Sales_Quantity"].sum().reset_index()

# Convert EV sales quantity to thousands (K) for better readability
sales_summary["EV_Sales_Quantity_K"] = sales_summary["EV_Sales_Quantity"] / 1000  # Convert to '000s

# Plot updated EV Sales Trend Over the Years
plt.figure(figsize=(12, 6))
sns.lineplot(data=sales_summary, x="Year", y="EV_Sales_Quantity_K", marker="o", color="b")

# Update labels and title
plt.xlabel("Year")
plt.ylabel("Total EV Sales (in Thousands)")
plt.title("Yearly Trend of EV Sales in India (Scaled)")
plt.grid(True)

# Show updated plot
plt.show()


# ---- Top 10 States by EV Sales ----
statewise_sales = df.groupby("State")["EV_Sales_Quantity"].sum().reset_index()
top_states = statewise_sales.sort_values(by="EV_Sales_Quantity", ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(data=top_states, x="EV_Sales_Quantity", y="State", palette="Blues_r")
plt.xlabel("Total EV Sales")
plt.ylabel("State")
plt.title("Top 10 States by EV Sales in India")
plt.grid(axis="x")
plt.show()





# ========================================
# STEP 4: FEATURE ENGINEERING
# ========================================

# Extract Month and Day from Date column
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# One-hot encoding for categorical variables
df_encoded = pd.get_dummies(df, columns=['State', 'Vehicle_Class', 'Vehicle_Category', 'Vehicle_Type'], drop_first=True)

# Drop unnecessary columns
df_encoded.drop(['Date', 'Month_Name'], axis=1, inplace=True)





# ========================================
# STEP 5: MODEL TRAINING
# ========================================

# Define features and target variable
X = df_encoded.drop('EV_Sales_Quantity', axis=1)
y = df_encoded['EV_Sales_Quantity']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)





# ========================================
# STEP 6: MODEL EVALUATION
# ========================================

# Calculate RMSE (Root Mean Squared Error)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'\nRoot Mean Squared Error (RMSE): {rmse:.2f}')

# ---- Actual vs Predicted EV Sales ----
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color="purple")
plt.xlabel('Actual EV Sales')
plt.ylabel('Predicted EV Sales')
plt.title('Actual vs Predicted EV Sales')
plt.grid(True)
plt.show()

# ---- Feature Importance ----
feature_importance = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)

plt.figure(figsize=(12, 6))
feature_importance.head(10).plot(kind='bar', color="teal")
plt.title('Top 10 Important Features in EV Sales Prediction')
plt.ylabel('Importance Score')
plt.xlabel('Features')
plt.show()





# ========================================
# STEP 7: CONCLUSION
# ========================================

print("\nKey Insights from the Analysis:\n")
print("- EV Sales have grown significantly over the years, especially after 2016.")
print("- Certain states dominate the EV market, while others lag behind.")
print("- The model helps understand which factors influence EV sales the most.")
print("- Feature importance analysis shows that some factors are more critical in predicting sales.")
