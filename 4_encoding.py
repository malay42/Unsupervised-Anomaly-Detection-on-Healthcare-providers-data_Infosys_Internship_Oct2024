# Encoding

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Load cleaned data
data = pd.read_csv('cleaned_data.csv')

# Columns for processing
numerical_cols = [
    'Number of Services', 'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services',
    'Average Medicare Allowed Amount', 'Average Submitted Charge Amount',
    'Average Medicare Payment Amount', 'Average Medicare Standardized Amount'
]

categorical_cols = [
    'Gender of the Provider', 'Entity Type of the Provider', 'HCPCS Drug Indicator',
    'Country Code of the Provider', 'Medicare Participation Indicator', 'Place of Service'
]

# Step 1: Visualize initial distributions of numerical columns before scaling
plt.figure(figsize=(14, 10))
for i, col in enumerate(numerical_cols, 1):
    if col in data.columns:  # Check if the column exists
        plt.subplot(3, 3, i)
        sns.histplot(data[col], kde=True, color='blue')
        plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

# Scale numerical data directly in the original DataFrame
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Encode categorical columns directly in the original DataFrame
encoder = OneHotEncoder(sparse_output=False, drop='first')  # Corrected parameter name
encoded_data = encoder.fit_transform(data[categorical_cols])
encoded_columns = encoder.get_feature_names_out(categorical_cols)

# Create a DataFrame of encoded columns
encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns, index=data.index)

# Drop original categorical columns and merge encoded columns
data.drop(columns=categorical_cols, inplace=True)
data = pd.concat([data, encoded_df], axis=1)

# Save changes to the same dataset
data.to_csv('cleaned_data.csv', index=False)
print("Scaling and encoding completed. Data saved in 'cleaned_data.csv'.")

# Step 2: Visualize distributions of numerical columns after scaling
plt.figure(figsize=(14, 10))
for i, col in enumerate(numerical_cols, 1):
    if col in data.columns:  # Ensure the column exists
        plt.subplot(3, 3, i)
        sns.histplot(data[col], kde=True, color='orange')
        plt.title(f"Scaled Distribution of {col}")
plt.tight_layout()
plt.show()
