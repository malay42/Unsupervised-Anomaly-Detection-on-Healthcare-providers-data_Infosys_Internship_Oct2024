import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv('Healthcare Providers.csv')
print(df)
print(df.info())
print(df.isnull().sum())


df.drop(columns=[
    'Middle Initial of the Provider', 
    'First Name of the Provider',
    'HCPCS Description',
    'Last Name/Organization Name of the Provider',
    'Street Address 1 of the Provider',
    'Street Address 2 of the Provider'
], inplace=True)

df.fillna({
    'Credentials of the Provider': 'Not Provided', 
    'Gender of the Provider': 'Unknown'
}, inplace=True)
print(df.isnull().sum())

numeric_columns = ['Number of Services', 'Number of Medicare Beneficiaries', 
        'Number of Distinct Medicare Beneficiary/Per Day Services', 
        'Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 
        'Average Medicare Payment Amount', 'Average Medicare Standardized Amount']

def clean_column(col):
    col = col.replace(',', '', regex=True)         # Remove commas
    col = pd.to_numeric(col, errors='coerce')      # Convert to numeric, coercing errors to NaN
    col = col.fillna(col.median())                 # Fill NaN with median
    return col.round(2)                            # Round to 2 decimal places

# Apply the cleaning function to each column in numeric_columns
df[numeric_columns] = df[numeric_columns].apply(clean_column)

print(df.dtypes)
df.to_csv('Cleaned_Healthcare Providers.csv', index= False)


# Selecting categorical columns (object and category types)
categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
print(categorical_columns)

# Univariate analysis
# Set the style for the plots
sns.set(style='whitegrid')
# Set up the figure for a grid layout of boxplots
plt.figure(figsize=(15, 10))
# Loop through columns and create boxplots
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot - {col}')

plt.tight_layout()
plt.show()

# Set up the number of rows and columns for subplots
n_cols = 3  # Number of columns per row
n_rows = (len(numeric_columns) + n_cols - 1) // n_cols  # Calculate number of rows

# Create a single figure with subplots
plt.figure(figsize=(18, 5 * n_rows))

# Loop through each numerical column and create a subplot for its histogram
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.histplot(df[col], bins=30, kde=True, color='blue', stat='density')
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Density')

# Add extra space between rows and columns
plt.subplots_adjust(hspace=0.6, wspace=0.3)
plt.show()

#Bivariate analysis
sns.set(style='whitegrid')
# Create boxplots for each numerical column grouped by each categorical column
for cat_col in categorical_columns:
    for num_col in numeric_columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=cat_col, y=num_col, data=df)
        plt.title(f'Boxplot of {num_col} by {cat_col}')
        plt.xticks(rotation=90)  # Rotate x labels if needed
        plt.ylabel(num_col)
        plt.xlabel(cat_col)
        plt.show()


#Heatmap
plt.figure(figsize=(12, 8))
corr = df[['Number of Services', 'Number of Medicare Beneficiaries', 
             'Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 
             'Average Medicare Payment Amount', 'Average Medicare Standardized Amount']].corr()

sns.heatmap(corr, annot=True,fmt=".2f", cmap='coolwarm', linewidths=0.5,square=True,cbar_kws={"shrink": .8})
plt.title('Correlation Heatmap')
plt.show()