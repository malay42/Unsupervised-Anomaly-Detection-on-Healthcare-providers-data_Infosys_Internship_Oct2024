# Healthcare Provider Data Cleaning Script
# This script performs data cleaning on the Healthcare Providers dataset
# focusing on handling missing values, removing duplicates, and preparing the data for analysis.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("C:\\Users\\malla\\Downloads\\archive\\Healthcare Providers.csv")

# 1. Drop columns with excessive missing values (threshold: >50% missing values)
missing_threshold = 0.5
df = df.loc[:, df.isnull().mean() < missing_threshold]

# 2. Handle missing values in categorical columns by imputing the most frequent value (mode)
categorical_columns = df.select_dtypes(include=['object', 'category']).columns
for col in categorical_columns:
most_frequent = df[col].mode()[0]
df[col] = df[col].fillna(most_frequent)

# 3. Handle missing values in numerical columns by imputing the median value
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_columns:
median_value = df[col].median()
df[col] = df[col].fillna(median_value)

# 4. Drop rows with any remaining NaN values (if necessary)
df = df.dropna()

# 5. Remove duplicate rows to ensure unique records
df = df.drop_duplicates()

# Final Check: Print remaining missing values per column to verify cleaning
print("Remaining missing values per column:")
print(df.isnull().sum())

# Save the cleaned dataset (Optional)
# df.to_csv("Healthcare_Providers_Cleaned.csv", index=False)
