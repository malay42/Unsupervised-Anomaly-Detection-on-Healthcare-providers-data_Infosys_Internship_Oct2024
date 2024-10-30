import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Healthcare_Providers.csv')        # Loading dataset

# Low Relevance Features
numeric_columns_median = ['Average Medicare Allowed Amount',
                'Average Medicare Standardized Amount',
                'Number of Medicare Beneficiaries',
                'Number of Distinct Medicare Beneficiary/Per Day Services']

# Function to remove outliers and replace them with median
def replace_outliers_with_median(df, col):
    Q1 = df[col].quantile(0.25)             # first quartile
    Q3 = df[col].quantile(0.75)             # third quartile
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    median_value = df[col].median()         # Replace outliers with the median value
    df[col] = df[col].apply(lambda x: median_value if x < lower_bound or x > upper_bound else x)
    
    return df

# Apply outlier removal for numeric columns with median replacement strategy
for col in numeric_columns_median:
    df = replace_outliers_with_median(df, col)

# visualizing after removal of outliers 
for col in numeric_columns_median:
    plt.figure(figsize=(5, 3))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot for {col}")
    plt.show()

# High Relevance Features - Outliers are kept as it is in this features as they may help in anomaly detection
high_rel_feat = ['Number of Services', 
                 'Average Submitted Charge Amount',
                 'Average Medicare Payment Amount']