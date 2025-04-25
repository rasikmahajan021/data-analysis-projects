# ========================================
# STEP 1: IMPORT LIBRARIES & LOAD DATASET
# ========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = "Supermart Grocery Sales - Retail Analytics Dataset.csv"
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

# Drop any rows with missing values
df.dropna(inplace=True)

# Convert 'Order Date' to datetime format
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')

# Extract Day, Month, and Year from 'Order Date'
df['Order Day'] = df['Order Date'].dt.day
df['Order Month'] = df['Order Date'].dt.month
df['Order Year'] = df['Order Date'].dt.year

# Drop duplicate rows
df.drop_duplicates(inplace=True)





# ========================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ========================================

# ---- Total Sales Over Time ----
plt.figure(figsize=(12, 6))
df.groupby('Order Date')['Sales'].sum().plot()
plt.title('Total Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.grid(True)
plt.show()

# ---- Top 10 Product Categories by Sales ----
category_sales = df.groupby("Category")["Sales"].sum().reset_index()
top_categories = category_sales.sort_values(by="Sales", ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(data=top_categories, x="Sales", y="Category", palette="Blues_r")
plt.xlabel("Total Sales")
plt.ylabel("Category")
plt.title("Top 10 Product Categories by Sales")
plt.grid(axis="x")
plt.show()


# Drop non-numeric columns before correlation analysis
df_numeric = df.select_dtypes(include=[np.number])

plt.figure(figsize=(12, 6))
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()





# ========================================
# STEP 4: FEATURE ENGINEERING
# ========================================

# Ensure 'Month' column exists before encoding
if 'Month' not in df.columns:
    df['Month'] = df['Order Date'].dt.strftime('%B')  # Extract full month name

# Encode categorical variables using Label Encoding
label_cols = ['Category', 'Sub Category', 'City', 'Region', 'State', 'Month']
le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])

# Drop unnecessary columns
df.drop(columns=['Order ID', 'Customer Name', 'Order Date'], inplace=True)





# ========================================
# STEP 5: MODEL TRAINING
# ========================================

# Check for missing values again after encoding
print("\nMissing Values After Encoding:\n")
print(df.isnull().sum())

# Fill missing values with the median for numerical columns
df.fillna(df.median(numeric_only=True), inplace=True)

# Define features and target variable
X = df.drop(columns=['Sales'])
y = df['Sales']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)






# ========================================
# STEP 6: MODEL EVALUATION
# ========================================

# Calculate MSE and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'\nMean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')

# ---- Actual vs Predicted Sales ----
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color="purple")
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.grid(True)
plt.show()




# ========================================
# STEP 7: CONCLUSION
# ========================================

print("\nKey Insights from the Analysis:\n")
print("- The model predicts sales reasonably well but may be improved with advanced models.")
print("- Sales have seasonal trends, which could be explored further.")
print("- Certain categories contribute significantly more to overall sales.")
