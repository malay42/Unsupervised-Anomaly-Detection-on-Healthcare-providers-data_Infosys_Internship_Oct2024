import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Load the data
file_path = "cleaned.csv"
df = pd.read_csv(file_path)

# Define numeric columns
numeric_columns = [
    'Number of Services', 'Number of Medicare Beneficiaries', 
    'Number of Distinct Medicare Beneficiary/Per Day Services',
    'Average Medicare Allowed Amount', 'Average Submitted Charge Amount',
    'Average Medicare Payment Amount', 'Average Medicare Standardized Amount'
]

# Step 1: Convert numeric columns to float, handling non-numeric values
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    missing_count = df[col].isna().sum()
    if missing_count > 0:
        df[col].fillna(df[col].median(), inplace=True)  # Fill NaN with the median
        print(f"Filled {missing_count} missing values in '{col}' with median.")

# Step 2: Detect and visualize outliers using Z-score
z_outliers = {}
plt.figure(figsize=(16, 12))
for i, col in enumerate(numeric_columns, 1):
    # Calculate Z-scores for each value in the column
    z_scores = stats.zscore(df[col])
    z_outliers[col] = np.where(np.abs(z_scores) > 3)[0]
    
    # Plot with outliers marked
    plt.subplot(3, 3, i)
    sns.boxplot(x=df[col], color='lightblue')
    plt.scatter(df.loc[z_outliers[col], col], [0] * len(z_outliers[col]), color='red', marker='x', label='Z-score Outliers')
    plt.title(f'{col} - Z-score Outliers')
    plt.legend()

plt.tight_layout()
plt.show()

# Step 3: Detect and visualize outliers using IQR
iqr_outliers = {}
plt.figure(figsize=(16, 12))
for i, col in enumerate(numeric_columns, 1):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    iqr_outliers[col] = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))].index
    
    # Plot with outliers marked
    plt.subplot(3, 3, i)
    sns.boxplot(x=df[col], color='lightgreen')
    plt.scatter(df.loc[iqr_outliers[col], col], [0] * len(iqr_outliers[col]), color='red', marker='x', label='IQR Outliers')
    plt.title(f'{col} - IQR Outliers')
    plt.legend()

plt.tight_layout()
plt.show()

# Step 4: Remove outliers using IQR
for col in numeric_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    original_count = df.shape[0]
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    removed_count = original_count - df.shape[0]
    print(f"Removed {removed_count} outliers from '{col}' using IQR method.")

# Step 5: Visualize cleaned data without outliers
plt.figure(figsize=(16, 12))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x=df[col], color='skyblue')
    plt.title(f'{col} - After Removing Outliers')

plt.tight_layout()
plt.show()

# Step 6: Save the cleaned data without outliers
output_file_path = "outliers.csv"
df.to_csv(output_file_path, index=False)
print(f"Cleaned data without outliers saved to {output_file_path}")
