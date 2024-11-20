# Cleaning

import pandas as pd


data = pd.read_csv('Healthcare Providers.csv')

# Columns for processing
numerical_cols = ['Number of Services', 'Number of Medicare Beneficiaries',
                  'Number of Distinct Medicare Beneficiary/Per Day Services',
                  'Average Medicare Allowed Amount', 'Average Submitted Charge Amount',
                  'Average Medicare Payment Amount', 'Average Medicare Standardized Amount']

categorical_cols = ['Gender of the Provider', 'Entity Type of the Provider', 'HCPCS Drug Indicator',
                    'Country Code of the Provider', 'Medicare Participation Indicator', 'HCPCS Code', 'Place of Service']

columns_to_drop = ['index', 'National Provider Identifier', 'Last Name/Organization Name of the Provider', 'First Name of the Provider',
       'Middle Initial of the Provider', 'Credentials of the Provider', 'Street Address 1 of the Provider', 'Street Address 2 of the Provider',
       'City of the Provider', 'Zip Code of the Provider', 'State Code of the Provider', 'Provider Type', 'HCPCS Description', 'HCPCS Code']

drop_col = data.drop(columns=columns_to_drop, errors='ignore', inplace=True)


null_values = data.isnull().sum()
print(null_values)
missing_percentage = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
print(f"\n\nPercentage of missing values: {missing_percentage:.2f}%")

# Clean numerical data
for col in numerical_cols:
    data[col] = data[col].str.replace(',', '').astype(float)

print(data.head())
print(data.info())

# Output cleaned data
data.to_csv('cleaned_data.csv', index=False)