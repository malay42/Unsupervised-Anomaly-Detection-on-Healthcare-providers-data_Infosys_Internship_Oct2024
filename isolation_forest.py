# Import required libraries
import pandas as pd
from sklearn.ensemble import IsolationForest

# Input and output file paths
INPUT_FILE = 'more_filtered_file.csv'
OUTPUT_FILE = 'isolation_new_file.csv'
df = pd.read_csv(INPUT_FILE)

# Numerical columns
NUMERICAL_COLUMNS = [
    'Number of Services',
    'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services',
    'Average Medicare Allowed Amount',
    'Average Submitted Charge Amount',
    'Average Medicare Payment Amount',
    'Average Medicare Standardized Amount'
]

# Clean numeric columns: removes whitespace and converts to numeric format
def clean_numeric_columns(df, columns):
    whitespace_count = 0
    for col in columns:
        # Count whitespace in column
        whitespace_count += df[col].astype(str).str.contains(r'^\s+|\s+$').sum()
        # Remove whitespace and convert to numeric
        df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce')
    return df, whitespace_count

# Clean the data
def clean_data(df):
    print("\n--- Data Cleaning ---")
    original_count = len(df)
    
    # Clean and convert numeric columns, count whitespace removals
    df, whitespace_count = clean_numeric_columns(df, NUMERICAL_COLUMNS)
    
    # Drop rows where all numerical columns are NaN and count NaN removals
    nan_before = df[NUMERICAL_COLUMNS].isna().sum().sum()
    df = df.dropna(subset=NUMERICAL_COLUMNS)
    nan_after = df[NUMERICAL_COLUMNS].isna().sum().sum()
    nan_cleared = nan_before - nan_after
    
    print(f"Original dataset count: {original_count}")
    print(f"Whitespace removed in numeric columns: {whitespace_count}")
    print(f"NaN values cleared: {nan_cleared}")
    print(f"Dataset size after cleaning: {df.shape}")
    return df

# Detect outliers
def detect_outliers(df, numeric_columns):
    print("\n--- Outlier Detection ---")
    # Select only numeric columns for Isolation Forest
    df_numeric = df[numeric_columns]
    
    # Initialize Isolation Forest
    iso_forest = IsolationForest(
        contamination=0.1,  # Assume 10% of the data points are outliers
        random_state=42,    
    )
    
    # Detect outliers
    outlier_labels = iso_forest.fit_predict(df_numeric)
    
    # Create mask for normal points (1) and outliers (-1)
    normal_data_mask = outlier_labels == 1
    
    # Remove outliers
    df_without_outliers = df[normal_data_mask].copy()
    num_outliers = len(df) - len(df_without_outliers)
    
    # Summary
    print(f"Dataset size after removing outliers: {len(df_without_outliers)} records")
    print(f"Number of outliers removed: {num_outliers}")
    print(f"Percentage of outliers: {(num_outliers / len(df) * 100):.2f}%")
    
    return df_without_outliers

# Processing the data
cleaned_df = clean_data(df)
final_df = detect_outliers(cleaned_df, NUMERICAL_COLUMNS)

# Save final cleaned dataset
print(f"\nSaving cleaned dataset to {OUTPUT_FILE}...")
final_df.to_csv(OUTPUT_FILE, index=False)
print("Processing complete!")
