import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Reading the data
df = pd.read_csv('Healthcare_Providers.csv')
df.isnull()  # checking for null values
df_clean = df.dropna()  # step 1: dropping null values
df_cleaned = df_clean.drop_duplicates()  # step 2: dropping duplicates

# Converting columns to numeric where necessary
numeric_columns = [
    'Zip Code of the Provider', 'Number of Services', 'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services', 'Average Medicare Allowed Amount',
    'Average Submitted Charge Amount', 'Average Medicare Payment Amount',
    'Average Medicare Standardized Amount'
]
for col in numeric_columns:
    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

def remove_outliers_iqr(df, column, multiplier=1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (multiplier * IQR)
    upper_bound = Q3 + (multiplier * IQR)
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

for col in numeric_columns:
    df_cleaned = remove_outliers_iqr(df_cleaned, col, multiplier=1.2)

plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(data=df_cleaned, y=col)
    plt.title(f"Boxplot of {col} (Stricter IQR)")
    plt.ylabel('Value')
plt.tight_layout()
plt.show()
