"""
Medicare Data Cleaning with IQR Method
removing missing values, whitespaces, 
and removing outliers.
"""

import pandas as pd

# Input and output files
input_file = 'more_filtered_file.csv'
output_file = 'iqr_cleaned_file.csv'

# clean numeric columns: removes whitespace and converts to numeric format
def clean_numeric_columns(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce')
    return df

# removing outliers using the IQR method
def remove_outliers_iqr(df, column):
    Q1, Q3 = df[column].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

# calculating outlier statistics
def iqr_outlier_summary(df, column):
    Q1, Q3 = df[column].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    return len(outliers), (len(outliers) / len(df[column])) * 100

# numeric columns to clean
numeric_columns = [
    'Number of Services',
    'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services',
    'Average Medicare Allowed Amount',
    'Average Submitted Charge Amount',
    'Average Medicare Payment Amount',
    'Average Medicare Standardized Amount'
]

# Loading data
data = pd.read_csv(input_file)
print(f"Initial data shape: {data.shape}")

# Clean data: remove whitespaces, convert to numeric, and drop rows with missing values
data = clean_numeric_columns(data, numeric_columns)
data = data.dropna(subset=numeric_columns)
print(f"Shape after cleaning missing values: {data.shape}")

# IQR outlier statistics for each numeric column
print("\nOutlier Analysis (IQR Method):")
for col in numeric_columns:
    outliers, percentage = iqr_outlier_summary(data, col)
    print(f"{col}: Outliers = {outliers}, Percentage = {percentage:.2f}%")

# Remove outliers for each numeric column
for col in numeric_columns:
    data = remove_outliers_iqr(data, col)

print(f"\nFinal data shape after removing outliers: {data.shape}")

# Save the cleaned data
data.to_csv(output_file, index=False)
print(f"\nCleaned data saved to: {output_file}")
