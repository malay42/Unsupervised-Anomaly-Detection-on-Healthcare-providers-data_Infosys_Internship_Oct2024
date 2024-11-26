import pandas as pd
import numpy as np

# numeric columns
numeric_columns = [
    'Number of Services',
    'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services',
    'Average Medicare Allowed Amount',
    'Average Submitted Charge Amount',
    'Average Medicare Payment Amount',
    'Average Medicare Standardized Amount'
]

# cleaning numeric data by removing commas
def clean_numeric_data(df):
    df_copy = df.copy()
    for col in numeric_columns:
        # Remove commas and convert to numeric
        df_copy[col] = pd.to_numeric(df_copy[col].astype(str).str.replace(',', ''), errors='coerce')
    return df_copy

# Load and clean all three data
original_df = clean_numeric_data(pd.read_csv('more_filtered_file.csv'))
iqr_df = clean_numeric_data(pd.read_csv('iqr_cleaned_file.csv'))
iso_df = clean_numeric_data(pd.read_csv('isolation_new_file.csv'))

# correlation comparison
def get_correlation_comparison(original_df, iqr_df, iso_df):
    print("Original correlations:")
    print(original_df[numeric_columns].corr().round(3))
    print("\nIQR correlations:")
    print(iqr_df[numeric_columns].corr().round(3))
    print("\nIsolation Forest correlations:")
    print(iso_df[numeric_columns].corr().round(3))

# statistical comparison
def get_comparison_stats(iqr_df, iso_df):
    for col in numeric_columns:
        print(f"\n=== Stats for {col} ===")
        print("\nIQR Method:")
        print(iqr_df[col].describe().round(3))
        print("\nIsolation Forest:")
        print(iso_df[col].describe().round(3))

# comparisons
print("=== CORRELATION COMPARISON ===")
get_correlation_comparison(original_df, iqr_df, iso_df)

print("\n=== STATISTICAL COMPARISON ===")
get_comparison_stats(iqr_df, iso_df)


