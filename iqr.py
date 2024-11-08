#importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt
#load the cleaned dataset
df = pd.read_csv('vis_data.csv')
#remove outliers using IQR
initial_size = df.shape[0]
def remove_outliers_iqr(df, column, multiplier=1.5):
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    # Calculate the Interquartile Range (IQR)
    IQR = Q3 - Q1
    # Define the lower and upper bounds
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    # Filter the DataFrame to remove outliers
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

# List of numeric columns after IQR outlier removal
numeric_columns = ['Number of Services', 'Average Medicare Allowed Amount', 
                   'Average Submitted Charge Amount', 'Average Medicare Payment Amount',
                   'Average Medicare Standardized Amount', 
                   'Number of Distinct Medicare Beneficiary/Per Day Services', 
                   'Number of Medicare Beneficiaries']
 

for col in numeric_columns:
    df_iqr = remove_outliers_iqr(df, col, multiplier=1.5)
iqr_size = df_iqr.shape[0]
iqr_percentage_removed = ((initial_size - iqr_size) / initial_size) * 100
print(f"Percentage of data removed by IQR: {iqr_percentage_removed:.2f}%")

# Function to visualize histograms and boxplots for each numeric column
def visualize_columns(df, numeric_columns):
    plt.figure(figsize=(15, 10))  # Adjust figure size to fit all graphs
    plt.suptitle('Data Distribution After IQR Outlier Removal', fontsize=20, fontweight='bold')
    
    # Iterate through each numeric column and create subplots
    for i, col in enumerate(numeric_columns, 1):
        plt.subplot(len(numeric_columns), 2, 2*i - 1)
        
        # Histogram with Kernel Density Estimate (KDE)
        sns.histplot(df[col], kde=True, bins=30, color='skyblue')
        plt.axvline(df[col].mean(), color='blue', linestyle='dashed', linewidth=2, label='Mean')
        plt.axvline(df[col].median(), color='red', linestyle='dashed', linewidth=2, label='Median')
        plt.legend()
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')

        # Boxplot to visualize outliers
        plt.subplot(len(numeric_columns), 2, 2*i)
        sns.boxplot(data=df, y=col, color='lightgreen')
        plt.title(f'Boxplot of {col}')
        plt.xlabel(col)
        plt.ylabel('Value')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit titles
    plt.show()

# Call the function to visualize each column after IQR outlier removal
visualize_columns(df, numeric_columns)
# Apply the IQR method with a stricter multiplier for outlier detection
for col in numeric_columns:
    df = remove_outliers_iqr(df, col, multiplier=1.2)  # Try using a lower multiplier like 1.2 or 1.0

# Re-plot the boxplots
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(data=df, y=col)
    plt.title(f"Boxplot of {col} (Stricter IQR)")
    plt.ylabel('Value')
plt.tight_layout()
plt.show()

# Summary statistics after IQR removal
print("Summary statistics after IQR outlier removal:")
print(df.describe())
print("Rows remaining after IQR removal:", len(df))

from scipy.stats import skew, kurtosis

# Example for one column after IQR  outlier removal
column = 'Average Medicare Allowed Amount'
print("Skewness after IQR:", skew(df[column]))
print("Kurtosis after IQR:", kurtosis(df[column]))
#Save the  data to a new CSV file
df.to_csv('iqr_data.csv', index=False)