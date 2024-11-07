#importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt
from scipy.stats import zscore
#load the cleaned dataset
df = pd.read_csv('vis_data.csv')
# outlier removal using z test 


# Function to remove outliers using Z-Score
def remove_outliers_zscore(df, column, threshold=3):
    """
    Remove outliers based on Z-Score method.
    :param df: Pandas DataFrame
    :param column: Column to remove outliers from
    :param threshold: Z-Score threshold (commonly 3)
    :return: DataFrame with outliers removed
    """
    # Ensure the column is numeric
    df[column] = pd.to_numeric(df[column], errors='coerce')
    
    # Drop rows with NaN values that result from conversion issues
    df = df.dropna(subset=[column])
    
    # Calculate Z-Scores for the column
    mean_col = df[column].mean()
    std_col = df[column].std()
    df['Z-Score'] = (df[column] - mean_col) / std_col
    
    # Filter the DataFrame to only keep rows where Z-Score is within the threshold
    df_no_outliers = df[(df['Z-Score'].abs() <= threshold)]
    
    # Drop the 'Z-Score' column as it's not needed anymore
    df_no_outliers = df_no_outliers.drop(columns=['Z-Score'])
    
    return df_no_outliers

# Apply the function to all numeric columns
numeric_columns = ['Number of Services', 'Average Medicare Allowed Amount', 
                   'Average Submitted Charge Amount', 'Average Medicare Payment Amount',
                   'Average Medicare Standardized Amount', 
                   'Number of Distinct Medicare Beneficiary/Per Day Services', 
                   'Number of Medicare Beneficiaries']

for col in numeric_columns:
    df = remove_outliers_zscore(df, col)

# Check the updated DataFrame
print(df.describe())


# List of numeric columns after Z-Score outlier removal
numeric_columns = ['Number of Services', 'Average Medicare Allowed Amount', 
                   'Average Submitted Charge Amount', 'Average Medicare Payment Amount',
                   'Average Medicare Standardized Amount', 
                   'Number of Distinct Medicare Beneficiary/Per Day Services', 
                   'Number of Medicare Beneficiaries']

# Function to visualize histograms and boxplots for each numeric column
def visualize_columns(df, numeric_columns):
    plt.figure(figsize=(15, 10))  # Adjust figure size to fit all graphs
    plt.suptitle('Data Distribution After Z-Score Outlier Removal', fontsize=20, fontweight='bold')
    
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

# Call the function to visualize each column after Z-Score outlier removal
visualize_columns(df, numeric_columns)

#box plot 

# Apply Z-score method to remove outliers
z_threshold = 2.5  # Adjust this threshold based on the desired strictness

# Filter the DataFrame to exclude rows where Z-scores exceed the threshold
df_z_filtered = df[(np.abs(zscore(df[numeric_columns])) < z_threshold).all(axis=1)]

# Re-plot the boxplots
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(data=df_z_filtered, y=col)
    plt.title(f"Boxplot of {col} (Z-score Threshold: {z_threshold})")
    plt.ylabel('Value')
plt.tight_layout()
plt.show()


# Summary statistics after Z-score removal
print("Summary statistics after Z-score outlier removal:")
print(df.describe())
print("Rows remaining after Z-score removal:", len(df))

from scipy.stats import skew, kurtosis
column = 'Average Medicare Allowed Amount'
print("Skewness after Z-score:", skew(df[column]))
print("Kurtosis after Z-score:", kurtosis(df[column]))

#Save the  data to a new CSV file
df.to_csv('z-score_data.csv', index=False)