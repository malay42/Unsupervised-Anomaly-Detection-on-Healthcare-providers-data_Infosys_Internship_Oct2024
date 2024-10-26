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
#df.to_csv('Cleaned_Healthcare Providers.csv', index= False)


# Selecting categorical columns (object and category types)
categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
print(categorical_columns)

# Univariate analysis
# Set the style for the plots
sns.set(style='whitegrid')
# Create histograms for all specified features
for col in numeric_columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[col], bins=30, kde=True, color='blue', stat='density')
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.axvline(df[col].mean(), color='red', linestyle='dashed', linewidth=1, label='Mean')
    plt.axvline(df[col].median(), color='green', linestyle='dashed', linewidth=1, label='Median')
    plt.legend()
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

sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
