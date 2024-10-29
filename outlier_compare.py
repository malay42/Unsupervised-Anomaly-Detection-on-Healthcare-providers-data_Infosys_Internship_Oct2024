import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

df = pd.read_csv('Healthcare_Providers.csv')
df_clean = df.dropna()  # Dropping null values
df_cleaned = df_clean.drop_duplicates()  # Dropping duplicates
numeric_columns = [
    'Zip Code of the Provider', 'Number of Services', 'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services', 'Average Medicare Allowed Amount',
    'Average Submitted Charge Amount', 'Average Medicare Payment Amount',
    'Average Medicare Standardized Amount'
]
for col in numeric_columns:
    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
df_cleaned = df_cleaned.dropna(subset=numeric_columns)
initial_size = df_cleaned.shape[0]
def remove_outliers_iqr(df, column, multiplier=1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (multiplier * IQR)
    upper_bound = Q3 + (multiplier * IQR)
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
for col in numeric_columns:
    df_cleaned = remove_outliers_iqr(df_cleaned, col, multiplier=1.2)
iqr_size = df_cleaned.shape[0]
iqr_percentage_removed = ((initial_size - iqr_size) / initial_size) * 100
print(f"Percentage of data removed by IQR: {iqr_percentage_removed:.2f}%")
df_cleaned_z = df_clean.drop_duplicates()  # fresh copy of data after nulls and duplicates removal
def remove_outliers_zscore(df, column, threshold=3):
    df[column] = pd.to_numeric(df[column], errors='coerce')
    df = df.dropna(subset=[column])
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores < threshold]
for col in numeric_columns:
    df_cleaned_z = remove_outliers_zscore(df_cleaned_z, col)
zscore_size = df_cleaned_z.shape[0]
zscore_percentage_removed = ((initial_size - zscore_size) / initial_size) * 100
print(f"Percentage of data removed by Z-score: {zscore_percentage_removed:.2f}%")

#encoding
columns_for_label_encoding = [
    'Gender of the Provider',
    'Entity Type of the Provider',
    'Medicare Participation Indicator',
    'HCPCS Drug Indicator',
    'Credentials of the Provider',  
    'HCPCS Code',  
    'HCPCS Description'
]

columns_for_one_hot_encoding = [
    'Provider Type',
    'Place of Service',
    'State Code of the Provider',
    'Country Code of the Provider'
]

encoder = LabelEncoder()

for column in columns_for_label_encoding:
    if column in df_cleaned.columns:
        df_cleaned[column] = encoder.fit_transform(df_cleaned[column])

df_cleaned = pd.get_dummies(df_cleaned, columns=columns_for_one_hot_encoding, drop_first=False)

df_cleaned = df_cleaned.replace({True: 'T', False: 'F'})

columns_to_normalize = [
    'Number of Services', 
    'Number of Medicare Beneficiaries', 
    'Number of Distinct Medicare Beneficiary/Per Day Services'
]

columns_to_standardize = [
    'Average Medicare Allowed Amount', 
    'Average Submitted Charge Amount', 
    'Average Medicare Payment Amount', 
    'Average Medicare Standardized Amount'
]

minmax_scaler = MinMaxScaler()
normalized_columns = minmax_scaler.fit_transform(df_cleaned[columns_to_normalize])

standard_scaler = StandardScaler()
standardized_columns = standard_scaler.fit_transform(df_cleaned[columns_to_standardize])

normalized_df = pd.DataFrame(normalized_columns, columns=columns_to_normalize)
standardized_df = pd.DataFrame(standardized_columns, columns=columns_to_standardize)

final_dataset = pd.concat([df_cleaned.drop(columns=columns_to_normalize + columns_to_standardize),
                           normalized_df, standardized_df], axis=1)

print(final_dataset.head())
