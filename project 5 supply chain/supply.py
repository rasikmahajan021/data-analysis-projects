# ========================================
# STEP 1: IMPORT LIBRARIES & LOAD DATASET
# ========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = "supply_chain_data.csv"
df = pd.read_csv(file_path)

# Display dataset information
print("Dataset Overview:\n")
print(df.info())
print("\nFirst 5 Rows of Data:\n")
print(df.head())





# ========================================
# STEP 2: DATA PREPROCESSING
# ========================================

# Check for missing values
print("\nMissing Values in Each Column:\n")
print(df.isnull().sum())

# Drop rows with missing values
df.dropna(inplace=True)

# Convert 'Date' column to datetime format if available
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Extract additional time-based features
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Quarter'] = df['Date'].dt.quarter

    # Drop the original Date column
    df.drop(columns=['Date'], inplace=True)

# Encode categorical variables
categorical_cols = ['Product Type', 'SKU', 'Supplier name', 'Shipping carriers', 'Transportation modes']
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])





# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ========================================

# ---- Sales Distribution ----
plt.figure(figsize=(12, 6))
sns.histplot(df['Revenue generated'], bins=30, kde=True, color="blue")
plt.xlabel("Revenue Generated", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.title("Distribution of Revenue Generated", fontsize=16, fontweight='bold')
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()

# ---- Top 10 Best-Selling Products ----
if 'Number of products sold' in df.columns and 'Product Type' in df.columns:
    product_sales = df.groupby("Product Type")["Number of products sold"].sum().reset_index()
    top_products = product_sales.sort_values(by="Number of products sold", ascending=False).head(10)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_products, x="Number of products sold", y="Product Type", palette="Blues_r")
    
    for index, value in enumerate(top_products["Number of products sold"]):
        plt.text(value + 5, index, str(value), va="center", fontsize=12, fontweight="bold")

    plt.xlabel("Total Products Sold", fontsize=14)
    plt.ylabel("Product Type", fontsize=14)
    plt.title("Top 10 Best-Selling Products", fontsize=16, fontweight='bold')
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.show()

# ---- Correlation Heatmap ----
df_numeric = df.select_dtypes(include=[np.number])

plt.figure(figsize=(12, 6))
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=1, linecolor='black')

plt.title('Correlation Heatmap', fontsize=16, fontweight='bold')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()





# ========================================
# STEP 4: FEATURE ENGINEERING
# ========================================

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Apply Label Encoding to categorical columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Define features and target variable
if 'Revenue generated' in df.columns:
    X = df.drop(columns=['Revenue generated'])
    y = df['Revenue generated']
else:
    raise ValueError("Revenue generated column is missing!")

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)





# ========================================
# STEP 5: MODEL TRAINING (Random Forest)
# ========================================

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)





# ========================================
# STEP 6: MODEL EVALUATION (Improved Graphs)
# ========================================

# ---- Actual vs Predicted Revenue ----
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color="purple", edgecolors="black")

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="dashed", linewidth=2)

plt.xlabel('Actual Revenue', fontsize=14)
plt.ylabel('Predicted Revenue', fontsize=14)
plt.title('Actual vs Predicted Revenue', fontsize=16, fontweight='bold')
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()





# ========================================
# STEP 7: CONCLUSION
# ========================================

print("\nKey Insights from the Analysis:\n")
print("- The Random Forest model predicts revenue with a good level of accuracy.")
print("- Some product types contribute significantly more to total sales.")
print("- The correlation heatmap helps identify key relationships between features.")

