import numpy as np
import pandas as pd

df = pd.read_csv("Healthcare.csv")

print(df.dtypes)
print(' ')

cols = ['Number of Services', 'Number of Medicare Beneficiaries', 'Number of Distinct Medicare Beneficiary/Per Day Services', 'Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 'Average Medicare Payment Amount', 'Average Medicare Standardized Amount']

for col in cols:
    df[col] = df[col].replace(',', '', regex= True)
    df[col] = pd.to_numeric(df[col], errors= 'coerce')
    df[col] = df[col].fillna(df[col].median())
    df[col] = df[col].round(2)

print(df.dtypes)
missing_values = df.isnull().sum()

print(missing_values[missing_values > 0])
print(' ')

# print(df['Gender of the Provider'].value_counts())
# print(df['Entity Type of the Provider'].value_counts())

df.drop(columns=['Middle Initial of the Provider', 'Street Address 2 of the Provider'], inplace= True)
missing_values = df.isnull().sum()

print(missing_values[missing_values > 0])
print(' ')

print(df['Credentials of the Provider'].value_counts())
df['Credentials of the Provider'] = df['Credentials of the Provider'].str.replace('.', '')

df['Credentials of the Provider']= df['Credentials of the Provider'].fillna(df['Credentials of the Provider'].mode()[0])

df['Number of Services'] = df['Number of Services'].astype(int)

df['Gender of the Provider'] = df['Gender of the Provider'].fillna(df['Gender of the Provider'].mode()[0])

df.fillna({'First Name of the Provider': 'Unknown'}, inplace= True)

missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])
print(' ')

duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")
df['Zip Code of the Provider'] = df['Zip Code of the Provider'].astype(str)
print(df.dtypes)
df.to_csv('Cleaned_HealthCare.csv', index= False)
