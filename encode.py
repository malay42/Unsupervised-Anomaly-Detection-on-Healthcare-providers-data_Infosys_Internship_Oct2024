#importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt

# Importing statistical tools and preprocessing modules
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
#load the cleaned dataset
df = pd.read_csv('iqr_data.csv')
#encoding
# List of categorical columns to apply label encoding (converting categories to numeric labels)
columns_for_label_encoding = [
    'Gender of the Provider',
    'Entity Type of the Provider',
    'Medicare Participation Indicator',
    'HCPCS Drug Indicator',
    'Credentials of the Provider',  
    'HCPCS Code',  
    'HCPCS Description'
]

# List of columns to apply one-hot encoding (creating dummy variables for categorical data)
columns_for_one_hot_encoding = [
    'Provider Type',
    'Place of Service',
    'State Code of the Provider',
    'Country Code of the Provider'
]
#encoder

# Initialize LabelEncoder for label encoding of categorical features

encoder = LabelEncoder()

# Applying label encoding to specified columns (convert string labels into numeric labels)
for column in columns_for_label_encoding:
    if column in df.columns:
        df[column] = encoder.fit_transform(df[column])

# Apply one-hot encoding to the specified columns, generating dummy variables
df = pd.get_dummies(df, columns=columns_for_one_hot_encoding, drop_first=False)

# Replace boolean values (True/False) with 'T'/'F' for readability

df = df.replace({True: 'T', False: 'F'})


# List of columns to normalize (scaling to range between 0 and 1)
columns_to_normalize = [
    'Number of Services', 
    'Number of Medicare Beneficiaries', 
    'Number of Distinct Medicare Beneficiary/Per Day Services'
]


# List of columns to standardize (scaling to have zero mean and unit variance)

columns_to_standardize = [
    'Average Medicare Allowed Amount', 
    'Average Submitted Charge Amount', 
    'Average Medicare Payment Amount', 
    'Average Medicare Standardized Amount'
]


# Initialize MinMaxScaler for normalization (scaling values to be between 0 and 1)
minmax_scaler = MinMaxScaler()

# Applying normalization to specified columns
normalized_columns = minmax_scaler.fit_transform(df[columns_to_normalize])
# Initialize StandardScaler for standardization (scaling values to have mean=0 and variance=1)

standard_scaler = StandardScaler()
# Applying standardization to specified columns

standardized_columns = standard_scaler.fit_transform(df[columns_to_standardize])
# Converting normalized and standardized data into DataFrames to retain column names

normalized_df = pd.DataFrame(normalized_columns, columns=columns_to_normalize)
standardized_df = pd.DataFrame(standardized_columns, columns=columns_to_standardize)

# Concatenating the normalized and standardized data with the original DataFrame, 
# removing the original columns that were normalized or standardized
final_dataset = pd.concat([df.drop(columns=columns_to_normalize + columns_to_standardize),
                           normalized_df, standardized_df], axis=1)

print(final_dataset.head())
# Save the final dataset to a CSV file
final_dataset.to_csv('final_dataset.csv', index=False)
