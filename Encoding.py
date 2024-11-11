import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load the cleaned data
file_path = "outliers.csv"  # Ensure the correct path to your cleaned dataset
df = pd.read_csv(file_path)

# Step 1: Binary Encoding for specified columns
binary_encoding_columns = ['Gender of the Provider', 'Entity Type of the Provider', 'Country Code of the Provider',
                           'Place of Service', 'Medicare Participation Indicator', 'HCPCS Drug Indicator']
binary_encoder = ce.BinaryEncoder(cols=binary_encoding_columns)
df_encoded = binary_encoder.fit_transform(df)

# Save the encoded dataset
encoded_file_path = "encoded_data.csv"
df_encoded.to_csv(encoded_file_path, index=False)
print(f"Encoded data saved to {encoded_file_path}")

# Display the first few rows to verify encoding
print(df_encoded.head())

# Define numeric columns for normalization
numeric_columns = [
    'Number of Services', 'Number of Medicare Beneficiaries', 
    'Number of Distinct Medicare Beneficiary/Per Day Services',
    'Average Medicare Allowed Amount', 'Average Submitted Charge Amount',
    'Average Medicare Payment Amount', 'Average Medicare Standardized Amount'
]

# Step 2: Plot initial distributions for each numeric column before normalization
plt.figure(figsize=(14, 10))
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df_encoded[column], kde=True, color='blue')
    plt.title(f"Distribution of {column}")
plt.tight_layout()
plt.show()

# Step 3: Initialize the MinMaxScaler and fit it to the data
scaler = MinMaxScaler()
scaler.fit(df_encoded[numeric_columns])  # Fit the scaler to the numeric columns

# Step 4: Transform the data using the fitted scaler
df_normalized = df_encoded.copy()
df_normalized[numeric_columns] = scaler.transform(df_encoded[numeric_columns])  # Apply the transformation

# Step 5: Plot distributions after normalization
plt.figure(figsize=(14, 10))
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df_normalized[column], kde=True, color='green')
    plt.title(f"Normalized {column}")
plt.tight_layout()
plt.show()

# Step 6: Save the normalized dataset
normalized_file_path = "normalized_data.csv"
df_normalized.to_csv(normalized_file_path, index=False)
print(f"Normalized dataset saved to {normalized_file_path}")
