import numpy as np
import pandas as pd 

df = pd.read_csv('Healthcare_Providers.csv')        # Loading dataset

# Dropping Unwanted Columns 
columns_to_drop = ['index',                                             # Irrelevant
                   'National Provider Identifier',                      # All unique values - not useful
                   'Last Name/Organization Name of the Provider',       # Last names of individuals are not relevant and org names present are only 0.02%
                   'First Name of the Provider',                        # First names are not relevant for anomaly detection tasks
                   'Middle Initial of the Provider',                    # It's a minor personal detail with limited analytical value
                   'Street Address 1 of the Provider',                  # Likely irrelevant 
                   'Street Address 2 of the Provider',                  # High missing values
                   'Zip Code of the Provider',                          # 50k unique values(wont help to detect anomaly)
                   'Medicare Participation Indicator',                  # about 99.9% entries are 'yes'
                   'HCPCS Description',                                 # HCPCS Code and description represent same thing 
                   'Country Code of the Provider',                      # about 99.9% entries are from US (only 6 entries are from different countries)
                   'Number of Distinct Medicare Beneficiary/Per Day Services',
                   'Average Medicare Payment Amount', 'Average Medicare Standardized Amount']                                        
df = df.drop(columns=columns_to_drop)

# Handling Missing Values 
print(f"Null Values \n{df.isnull().sum()}")
null_values = ['Credentials of the Provider',               # Null values are present only in 2 features
               'Gender of the Provider']
print(f"Percentage of null Values: \n{df[null_values].isnull().sum() / df.shape[0] * 100}")

constant = 'org'        # Both columns are categorical - simply fill with 'org' as blank space indicates that it is an organization
df["Credentials of the Provider"].replace(np.nan, constant, inplace = True)
df["Gender of the Provider"].replace(np.nan, constant, inplace = True)

print(df[null_values].isnull().sum())       # Rechecking whether null values are present

# Converting Data Types - Categorical to Numerical
df['Number of Services'] = df['Number of Services'].str.replace(',', '').astype(float)
df['Number of Medicare Beneficiaries'] = df['Number of Medicare Beneficiaries'].str.replace(',', '').astype(float)
df['Average Medicare Allowed Amount'] = df['Average Medicare Allowed Amount'].str.replace(',', '').astype(float)
df['Average Submitted Charge Amount'] = df['Average Submitted Charge Amount'].str.replace(',', '').astype(float)

# Standarizing Column Values 
obj_features = df.select_dtypes(include=["object"]).columns
print("Total number categorical of features: ", obj_features.shape[0])
print("Categorical features name: ", obj_features.tolist())

df[obj_features] = df[obj_features].apply(lambda x: x.str.lower())      # Convert categorical columns to lowercase 
df[obj_features] = df[obj_features].apply(lambda x: x.str.strip())      # Strip leading/trailing spaces

# Consistent Categories - For columns like Credentials, ensure consistency (e.g., MD, M.D., md should all be standardized)
for col in obj_features:                        # Print unique values for each categorical column to identify inconsistencies
    unique_values = df[col].unique()
    print(f"Unique values in '{col}' column:")
    print(unique_values)
    print("\n")

df['Credentials of the Provider'] = df['Credentials of the Provider'].replace(['m.d.', 'md', 'MD'], 'md')
df['Credentials of the Provider'] = df['Credentials of the Provider'].replace(['d.o.', 'do', 'DO'], 'do')