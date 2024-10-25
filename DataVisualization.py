import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('Healthcare Providers.csv')

numeric_columns = ['Number of Services', 'Number of Medicare Beneficiaries',
                   'Number of Distinct Medicare Beneficiary/Per Day Services',
                   'Average Medicare Allowed Amount', 'Average Submitted Charge Amount',
                   'Average Medicare Payment Amount', 'Average Medicare Standardized Amount']

#Bivariate Analysis

#Heatmap no numerical column
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')

plt.figure(figsize=(10, 8))
correlation_matrix = df[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

#Boxplot between Gender of the provider and Average Medical Payment Amount
plt.figure(figsize=(12, 8))
sns.boxplot(data=df, x='Gender of the Provider', y='Average Medicare Payment Amount')
plt.title('Boxplot of Medicare Payment by Gender')
plt.show()

#Scatter plot between Average Medicare Payment Amount and Number of Services
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Average Medicare Payment Amount', y='Number of Services', alpha=0.7)
plt.title('Scatter Plot of Average Medicare Payment Amount vs. Number of Services')
plt.xlabel('Average Medicare Payment Amount')
plt.ylabel('Number of Services')
plt.show()

#Univariate Analysis

#Pair Plot of Numeric Features
sns.pairplot(df[numeric_columns])
plt.title('Pair Plot of Numeric Features')
plt.show()

#Histogram agains the frequency of each Numerical columns
for col in numeric_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

#boxplot for each numerical columns
for col in numeric_columns:
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x=col)
    plt.title('Boxplot of Number of Services')
    plt.show()
