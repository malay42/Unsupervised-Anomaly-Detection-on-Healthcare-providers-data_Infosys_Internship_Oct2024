# Importing necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#load the dataset
df = pd.read_csv('Cleaned_Healthcare_Providers.csv')



# Define numeric columns for analysis
numeric_columns = [
    'Number of Services', 
    'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services',
    'Average Medicare Allowed Amount', 
    'Average Submitted Charge Amount',
    'Average Medicare Payment Amount', 
    'Average Medicare Standardized Amount'
]

# Convert numeric columns that are stored as strings with commas
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')

# Bivariate Analysis

# Heatmap of correlation among numeric columns
plt.figure(figsize=(10, 8))
correlation_matrix = df[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Boxplot between Gender of the provider and Average Medicare Payment Amount
plt.figure(figsize=(12, 8))
sns.boxplot(data=df, x='Gender of the Provider', y='Average Medicare Payment Amount')
plt.title('Boxplot of Medicare Payment by Gender')
plt.show()

# Scatter plot between Average Medicare Payment Amount and Number of Services
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Average Medicare Payment Amount', y='Number of Services', alpha=0.7)
plt.title('Scatter Plot of Average Medicare Payment Amount vs. Number of Services')
plt.xlabel('Average Medicare Payment Amount')
plt.ylabel('Number of Services')
plt.show()

# Univariate Analysis

# Pair Plot of Numeric Features
sns.pairplot(df[numeric_columns])
plt.suptitle('Pair Plot of Numeric Features', y=1.02)
plt.show()

# Histogram of each numeric column
for col in numeric_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# Boxplot for each numeric column
for col in numeric_columns:
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x=col)
    plt.title(f'Boxplot of {col}')
    plt.show()

