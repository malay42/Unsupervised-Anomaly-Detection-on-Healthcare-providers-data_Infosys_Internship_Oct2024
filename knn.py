import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors

# Load and clean the data
df = pd.read_csv('Healthcare_Providers.csv')
df_cleaned = df.drop_duplicates()

# Convert numeric columns to appropriate types
numeric_columns = [
    'Zip Code of the Provider', 'Number of Services', 'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services', 'Average Medicare Allowed Amount',
    'Average Submitted Charge Amount', 'Average Medicare Payment Amount',
    'Average Medicare Standardized Amount'
]
for col in numeric_columns:
    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

# Encode categorical columns
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

# Imputing missing values
imputer = SimpleImputer(strategy="median")
df_cleaned[numeric_columns] = imputer.fit_transform(df_cleaned[numeric_columns])

# Normalize and standardize selected columns
columns_to_normalize = [
    'Number of Services', 'Number of Medicare Beneficiaries', 'Number of Distinct Medicare Beneficiary/Per Day Services'
]
columns_to_standardize = [
    'Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 
    'Average Medicare Payment Amount', 'Average Medicare Standardized Amount'
]

# MinMax scaling and Standard scaling
minmax_scaler = MinMaxScaler()
normalized_df = pd.DataFrame(minmax_scaler.fit_transform(df_cleaned[columns_to_normalize]), columns=columns_to_normalize)

standard_scaler = StandardScaler()
standardized_df = pd.DataFrame(standard_scaler.fit_transform(df_cleaned[columns_to_standardize]), columns=columns_to_standardize)

# Combine into final dataset
final_dataset = pd.concat([df_cleaned.drop(columns=columns_to_normalize + columns_to_standardize),
                           normalized_df, standardized_df], axis=1)

# Apply K-Nearest Neighbors for anomaly detection
n_neighbors = 5
knn = NearestNeighbors(n_neighbors=n_neighbors)
knn.fit(final_dataset[numeric_columns])

# Calculate the average distance to the nearest neighbors
distances, _ = knn.kneighbors(final_dataset[numeric_columns])
avg_distance = distances.mean(axis=1)

# Set a threshold to identify anomalies
threshold = avg_distance.mean() + avg_distance.std()
final_dataset['anomaly_knn'] = (avg_distance > threshold).astype(int)

# Separate normal data and anomalies for KNN
anomalies_knn = final_dataset[final_dataset['anomaly_knn'] == 1]
normal_data_knn = final_dataset[final_dataset['anomaly_knn'] == 0]

# Display anomalies
print("Anomalies detected by KNN:")
print(anomalies_knn.head())
print(f"Total number of anomalies detected by KNN: {anomalies_knn.shape[0]}")

# Scatter plot of anomalies vs normal data for KNN
plt.figure(figsize=(10, 6))
plt.scatter(normal_data_knn['Number of Services'], normal_data_knn['Average Medicare Payment Amount'], 
            color='blue', label='Normal', alpha=0.5)
plt.scatter(anomalies_knn['Number of Services'], anomalies_knn['Average Medicare Payment Amount'], 
            color='green', label='Anomalies (KNN)', alpha=0.7)
plt.xlabel('Number of Services')
plt.ylabel('Average Medicare Payment Amount')
plt.title('Anomaly Detection with KNN')
plt.legend()
plt.show()
