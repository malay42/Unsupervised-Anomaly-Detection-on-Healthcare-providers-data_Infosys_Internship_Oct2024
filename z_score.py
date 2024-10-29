import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv('Healthcare_Providers.csv')
df_clean = df.dropna()  # Dropping null values
df_cleaned = df_clean.drop_duplicates()  # Dropping duplicates
numeric_columns = [
    'Zip Code of the Provider', 'Number of Services', 'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services', 'Average Medicare Allowed Amount',
    'Average Submitted Charge Amount', 'Average Medicare Payment Amount',
    'Average Medicare Standardized Amount'
]
for col in numeric_columns:
    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')  # Convert to numeric, invalid parsing will be set as NaN
df_cleaned = df_cleaned.dropna(subset=numeric_columns)
initial_size = df_cleaned.shape[0]
# Function to remove outliers using IQR
def remove_outliers_iqr(df, column, multiplier=1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (multiplier * IQR)
    upper_bound = Q3 + (multiplier * IQR)
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
for col in numeric_columns:
    df_cleaned = remove_outliers_iqr(df_cleaned, col, multiplier=1.2)
iqr_size = df_cleaned.shape[0]
iqr_percentage_removed = ((initial_size - iqr_size) / initial_size) * 100
print(f"Percentage of data removed by IQR: {iqr_percentage_removed:.2f}%")

plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(data=df_cleaned, y=col)
    plt.title(f"Boxplot of {col} (Stricter IQR)")
    plt.ylabel('Value')
plt.tight_layout()
plt.show()

# Re-reading data for Z-score method
df_cleaned_z = df_clean.drop_duplicates()  # fresh copy of data after nulls and duplicates removal
def remove_outliers_zscore(df, column, threshold=3):
    df[column] = pd.to_numeric(df[column], errors='coerce')
    df = df.dropna(subset=[column])
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores < threshold]
for col in numeric_columns:
    df_cleaned_z = remove_outliers_zscore(df_cleaned_z, col)
zscore_size = df_cleaned_z.shape[0]
zscore_percentage_removed = ((initial_size - zscore_size) / initial_size) * 100
print(f"Percentage of data removed by Z-score: {zscore_percentage_removed:.2f}%")

plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(data=df_cleaned_z, y=col)
    plt.title(f"Boxplot of {col} (Z-score)")
    plt.ylabel('Value')
plt.tight_layout()
plt.show()
