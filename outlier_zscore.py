"""
Medicare Data Cleaning Script using Z-Score Method
Handles missing values and removes outliers using the Z-score approach
"""
import pandas as pd
import numpy as np
from scipy import stats

# Input file path
input_file = 'more_filtered_file.csv'

# Cleaning numeric columns by removing whitespace and converting to numeric format.
def clean_numeric_columns(df, columns):
    cleaned_df = df.copy()
    for col in columns:
        cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
    return cleaned_df

# Remove outliers using Z-score method
def remove_outliers_zscore(df, column, threshold=3):
    mean = df[column].mean()
    std = df[column].std()
    df['z_score'] = (df[column] - mean) / std
    return df[np.abs(df['z_score']) <= threshold].drop(columns=['z_score'])

# Calculate number and percentage of outliers using Z-score method.
def zscore_outlier_summary(df, column, threshold=3):
    mean = df[column].mean()
    std = df[column].std()
    df['z_score'] = (df[column] - mean) / std
    outliers = df[(df['z_score'] > threshold) | (df['z_score'] < -threshold)]
    total_values = len(df[column])
    percentage = (len(outliers) / total_values) * 100
    df = df.drop(columns=['z_score'])
    return len(outliers), percentage

# columns that contain numeric values
numeric_columns = [
    'Number of Services',
    'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services',
    'Average Medicare Allowed Amount',
    'Average Submitted Charge Amount',
    'Average Medicare Payment Amount',
    'Average Medicare Standardized Amount'
]

# Load and display initial data info
data = pd.read_csv(input_file)
print(f"Initial data shape: {data.shape}")

# Clean numeric columns
cleaned_data = clean_numeric_columns(data, numeric_columns)

# Remove rows with missing values
cleaned_data = cleaned_data.dropna(subset=numeric_columns)
rows_removed = data.shape[0] - cleaned_data.shape[0]
print(f"\nRows removed due to missing values: {rows_removed}")
print(f"Shape after cleaning missing values: {cleaned_data.shape}")

# Calculate and display Z-score outlier statistics
print("\nAnalyzing outliers using Z-score method:")
for col in numeric_columns:
    outliers, percentage = zscore_outlier_summary(cleaned_data, col)
    print(f"Z-score - {col}: Number of Outliers: {outliers}, Percentage of Outliers: {percentage:.2f}%")

# Apply Z-score outlier removal to each column
data_no_outliers = cleaned_data.copy()
for col in numeric_columns:
    data_no_outliers = remove_outliers_zscore(data_no_outliers, col)

# Calculate total rows removed
rows_removed_outliers = cleaned_data.shape[0] - data_no_outliers.shape[0]
print(f"\nRows removed due to outliers: {rows_removed_outliers}")
print(f"Final data shape: {data_no_outliers.shape}")

# Define output file path and save cleaned dataset
output_file = 'zscore_cleaned_file.csv'
data_no_outliers.to_csv(output_file, index=False)
print(f"\nCleaned data saved to: {output_file}")
