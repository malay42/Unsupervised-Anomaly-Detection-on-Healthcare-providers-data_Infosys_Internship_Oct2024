import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Load and clean the data
df = pd.read_csv('Healthcare_Providers.csv')
df_cleaned = df.drop_duplicates()

# Convert numeric columns to numeric data types
numeric_columns = [
    'Zip Code of the Provider', 'Number of Services', 'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services', 'Average Medicare Allowed Amount',
    'Average Submitted Charge Amount', 'Average Medicare Payment Amount',
    'Average Medicare Standardized Amount'
]

for col in numeric_columns:
    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

# Label encode categorical columns
columns_for_label_encoding = [
    'Gender of the Provider', 'Entity Type of the Provider', 'Medicare Participation Indicator',
    'HCPCS Drug Indicator', 'Credentials of the Provider', 'HCPCS Code', 'HCPCS Description'
]
encoder = LabelEncoder()
for column in columns_for_label_encoding:
    if column in df_cleaned.columns:
        df_cleaned[column] = encoder.fit_transform(df_cleaned[column].astype(str))

# One-hot encode specified columns
columns_for_one_hot_encoding = [
    'Provider Type', 'Place of Service', 'State Code of the Provider', 'Country Code of the Provider'
]
df_cleaned = pd.get_dummies(df_cleaned, columns=columns_for_one_hot_encoding, drop_first=True)

# Impute missing values in numeric columns
imputer = SimpleImputer(strategy="median")
df_cleaned[numeric_columns] = imputer.fit_transform(df_cleaned[numeric_columns])

# Normalize and standardize specific columns
columns_to_normalize = [
    'Number of Services', 'Number of Medicare Beneficiaries', 'Number of Distinct Medicare Beneficiary/Per Day Services'
]
columns_to_standardize = [
    'Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 
    'Average Medicare Payment Amount', 'Average Medicare Standardized Amount'
]

minmax_scaler = MinMaxScaler()
normalized_df = pd.DataFrame(minmax_scaler.fit_transform(df_cleaned[columns_to_normalize]), columns=columns_to_normalize)

standard_scaler = StandardScaler()
standardized_df = pd.DataFrame(standard_scaler.fit_transform(df_cleaned[columns_to_standardize]), columns=columns_to_standardize)

# Combine the processed columns into final dataset
final_dataset = pd.concat([df_cleaned.drop(columns=columns_to_normalize + columns_to_standardize),
                           normalized_df, standardized_df], axis=1)

# Apply DBSCAN for anomaly detection
dbscan = DBSCAN(eps=0.5, min_samples=5)
final_dataset['anomaly'] = dbscan.fit_predict(final_dataset[numeric_columns])

# Count the anomalies
anomalies = final_dataset[final_dataset['anomaly'] == -1]
print("Anomalies detected by DBSCAN:")
print(anomalies.head())
print(f"Total number of anomalies detected by DBSCAN: {anomalies.shape[0]}")

# Scatter plot of Average Medicare Payment Amount vs Number of Medicare Beneficiaries
plt.figure(figsize=(10, 6))
plt.scatter(final_dataset['Average Medicare Payment Amount'], final_dataset['Number of Medicare Beneficiaries'],
            c=final_dataset['anomaly'], cmap='coolwarm', label='Anomalies')
plt.colorbar(label='Anomaly Status (1: Normal, -1: Anomaly)')
plt.xlabel('Average Medicare Payment Amount')
plt.ylabel('Number of Medicare Beneficiaries')
plt.title('Scatter Plot of Average Medicare Payment Amount vs Number of Medicare Beneficiaries')
plt.axhline(0, color='gray', lw=0.5)
plt.axvline(0, color='gray', lw=0.5)
plt.grid()
plt.legend()
plt.show()
