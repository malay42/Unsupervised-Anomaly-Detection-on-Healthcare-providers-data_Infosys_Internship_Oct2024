import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import category_encoders as ce
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
df = pd.read_csv('Healthcare Providers.csv')

# Define numeric and categorical columns
numeric_columns = ['Number of Services', 'Number of Medicare Beneficiaries',
                   'Number of Distinct Medicare Beneficiary/Per Day Services',
                   'Average Medicare Allowed Amount', 'Average Submitted Charge Amount',
                   'Average Medicare Payment Amount', 'Average Medicare Standardized Amount']

categorical_col = [
    'Credentials of the Provider', 'Gender of the Provider', 'Entity Type of the Provider',
    'City of the Provider', 'State Code of the Provider', 'Country Code of the Provider',
    'Provider Type', 'Medicare Participation Indicator', 'Place of Service', 'HCPCS Code',
    'HCPCS Drug Indicator'
]

# Display unique values in categorical columns
for col in categorical_col:
    unique_count = df[col].nunique()
    print(f"Number of unique values in '{col}': {unique_count}")

# Binary Encoding
binary_col = ['Gender of the Provider', 'Medicare Participation Indicator', 'Place of Service', 'HCPCS Drug Indicator']
binary_encoder = ce.BinaryEncoder(cols=binary_col)
binary_encoded = binary_encoder.fit_transform(df[binary_col])

# Plot Binary Encoded Data
plt.figure(figsize=(10, 6))
binary_encoded.plot(kind='hist', alpha=0.7, bins=5, ax=plt.gca())
plt.title('Binary Encoding Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# One-Hot Encoding
oneHot_col = ['Country Code of the Provider', 'State Code of the Provider']
oneHot_encoded = pd.get_dummies(df[oneHot_col], drop_first=True)

# Plot One-Hot Encoded Data
plt.figure(figsize=(10, 6))
oneHot_encoded.plot(kind='hist', alpha=0.7, bins=5, ax=plt.gca())
plt.title('One-Hot Encoding Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# Label Encoding
label_encode_columns = ['Credentials of the Provider', 'Provider Type']
label_encoded_data = {}
le = LabelEncoder()
for col in label_encode_columns:
    df[col] = df[col].fillna('Unknown')
    label_encoded_data[col] = le.fit_transform(df[col])

label_encoded = pd.DataFrame(label_encoded_data)

# Plot Label Encoded Data
plt.figure(figsize=(10, 6))
label_encoded.plot(kind='hist', alpha=0.7, bins=3, ax=plt.gca())
plt.title('Label Encoding Distribution')
plt.xlabel('Encoded Value')
plt.ylabel('Frequency')
plt.show()

# Frequency Encoding
freq_col = ['City of the Provider', 'HCPCS Code']
freq_encoded = pd.DataFrame()
for col in freq_col:
    frequency_encoded = df[col].value_counts().to_dict()
    freq_encoded[col + '_Encoded'] = df[col].map(frequency_encoded)

# Plot Frequency Encoded Data
plt.figure(figsize=(10, 6))
freq_encoded.plot(kind='hist', alpha=0.7, bins=5, ax=plt.gca())
plt.title('Frequency Encoding Distribution')
plt.xlabel('Encoded Frequency')
plt.ylabel('Frequency')
plt.show()

# Standardizing the numeric columns
numeric_col = df[numeric_columns]
scaler = StandardScaler()
scaled_numerical = pd.DataFrame(scaler.fit_transform(numeric_col), columns=numeric_col.columns)
scaled_numerical = scaled_numerical.round(2)

# Plot Scaled Numeric Data
plt.figure(figsize=(10, 6))
scaled_numerical.plot(kind='hist', alpha=0.7, bins=5, ax=plt.gca())
plt.title('Scaled Numeric Data Distribution')
plt.xlabel('Scaled Value')
plt.ylabel('Frequency')
plt.show()

# Reset index for all DataFrames before concatenation
binary_encoded.reset_index(drop=True, inplace=True)
oneHot_encoded.reset_index(drop=True, inplace=True)
label_encoded.reset_index(drop=True, inplace=True)
freq_encoded.reset_index(drop=True, inplace=True)
scaled_numerical.reset_index(drop=True, inplace=True)

# Concatenate all encoded and scaled data into a final DataFrame
final = pd.concat([scaled_numerical, binary_encoded, oneHot_encoded, label_encoded, freq_encoded], axis=1)
final.head()