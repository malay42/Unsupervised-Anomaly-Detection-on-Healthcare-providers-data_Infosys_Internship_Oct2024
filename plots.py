import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Reading the data
df=pd.read_csv('Healthcare_Providers.csv')
df.isnull() # checking for null values
df_clean = df.dropna() # step 1: dropping null values
df_cleaned = df_clean.drop_duplicates() # step 2: dropping duplicates
print(df_cleaned.dtypes) # Zip code here is float64

# converting all the numberic column to either int or float datatype so that there is no issue in plotting
df_cleaned['Zip Code of the Provider'] = pd.to_numeric(df_cleaned['Zip Code of the Provider'], errors='coerce')
df_cleaned['Number of Services'] = pd.to_numeric(df_cleaned['Number of Services'], errors='coerce')
df_cleaned['Number of Medicare Beneficiaries'] = pd.to_numeric(df_cleaned['Number of Medicare Beneficiaries'], errors='coerce')
df_cleaned['Number of Distinct Medicare Beneficiary/Per Day Services'] = pd.to_numeric(df_cleaned['Number of Distinct Medicare Beneficiary/Per Day Services'], errors='coerce')
df_cleaned['Average Medicare Allowed Amount'] = pd.to_numeric(df_cleaned['Average Medicare Allowed Amount'], errors='coerce')
df_cleaned['Average Submitted Charge Amount'] = pd.to_numeric(df_cleaned['Average Submitted Charge Amount'], errors='coerce')
df_cleaned['Average Medicare Payment Amount'] = pd.to_numeric(df_cleaned['Average Medicare Payment Amount'], errors='coerce')
df_cleaned['Average Medicare Standardized Amount'] = pd.to_numeric(df_cleaned['Average Medicare Standardized Amount'], errors='coerce') # Handling the data types
print(df_cleaned.dtypes)

# Count plot for Gender of the Provider
plt.figure(figsize=(8, 5))
sns.countplot(data=df_cleaned, x='Gender of the Provider')
plt.title('Count of Providers by Gender')
plt.xlabel('Gender of the Provider')
plt.ylabel('Count')
plt.show()

# Count plot for Provider Type
plt.figure(figsize=(12, 6))
sns.countplot(data=df_cleaned, y='Provider Type', order=df_cleaned['Provider Type'].value_counts().index)
plt.title('Count of Providers by Type')
plt.xlabel('Count')
plt.ylabel('Provider Type')
plt.show()


# All in one histogram
numerical_columns = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
numerical_columns = numerical_columns.drop(['Zip Code of the Provider', 'index', 'National Provider Identifier'])
print(df_cleaned[numerical_columns].describe())
categorical_columns = df_cleaned.select_dtypes(include=['object']).columns
for col in categorical_columns:
    print(f"Value counts for {col}:\n", df_cleaned[col].value_counts(), "\n")
df_cleaned[numerical_columns].hist(bins=15, figsize=(15, 10), layout=(4, 3))
plt.tight_layout()
for ax in plt.gcf().axes:
    ax.set_xlabel(ax.get_title())  # Set title as the X label
    ax.set_ylabel('Frequency')     # Set Y label as Frequency
plt.show()


# Boxplots for numerical variables to check for outliers
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(4, 3, i)
    sns.boxplot(data=df_cleaned, y=col)
    plt.title(f"Boxplot of {col}")
    plt.xlabel(col)
    plt.ylabel('Value')
plt.tight_layout()
plt.show()

# HeatMap for bivariate analysis
numerical_columns = [
    'Zip Code of the Provider', 'Number of Services', 
    'Number of Medicare Beneficiaries', 
    'Number of Distinct Medicare Beneficiary/Per Day Services',
    'Average Medicare Allowed Amount', 
    'Average Submitted Charge Amount', 
    'Average Medicare Payment Amount', 
    'Average Medicare Standardized Amount'
]
df_numerical = df_cleaned[numerical_columns]
correlation_matrix = df_numerical.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Heatmap of Correlation Matrix for Numerical Variables")
plt.xlabel("Numerical Variables")
plt.ylabel("Numerical Variables")
plt.show()