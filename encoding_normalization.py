import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
import category_encoders as ce

# Step 1: Read the CSV file
data = pd.read_csv('iqr_cleaned_file.csv')

# Step 2: Define numeric and categorical columns based on actual column names
number_columns = [
    'Number of Services',
    'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services',
    'Average Medicare Allowed Amount',
    'Average Submitted Charge Amount',
    'Average Medicare Payment Amount',
    'Average Medicare Standardized Amount'
]

text_columns = [
    'Last Name/Organization Name of the Provider',
    'First Name of the Provider',
    'Gender of the Provider',
    'Entity Type of the Provider',
    'State Code of the Provider',
    'Country Code of the Provider',
    'Provider Type',
    'Medicare Participation Indicator',
    'Place of Service',
    'HCPCS Code',
    'HCPCS Description',
    'HCPCS Drug Indicator'
]

# Step 3: Display unique values in each categorical column
for column in text_columns:
    unique_values = data[column].nunique()
    print(f"Column '{column}' has {unique_values} different values")

# Step 4: Binary Encoding (for Yes/No type columns)
yes_no_columns = [
    'Gender of the Provider',
    'Medicare Participation Indicator',
    'Place of Service',
    'HCPCS Drug Indicator'
]
binary_tool = ce.BinaryEncoder(cols=yes_no_columns)
binary_data = binary_tool.fit_transform(data[yes_no_columns])

# Step 5: One-Hot Encoding (for location-based columns)
location_columns = ['Country Code of the Provider', 'State Code of the Provider']
location_data = pd.get_dummies(data[location_columns], drop_first=True)

# Step 6: Label Encoding (for provider-related information)
provider_columns = [
    'Last Name/Organization Name of the Provider',
    'First Name of the Provider',
    'Provider Type'
]
label_tool = LabelEncoder()
provider_data = pd.DataFrame()

for column in provider_columns:
    # Fill missing values with 'Unknown' before encoding
    filled_column = data[column].fillna('Unknown')
    provider_data[column] = label_tool.fit_transform(filled_column)

# Step 7: Frequency Encoding (for description and code columns)
count_columns = ['HCPCS Code', 'HCPCS Description']
count_data = pd.DataFrame()

for column in count_columns:
    value_counts = data[column].value_counts()
    count_data[column] = data[column].map(value_counts)

# Step 8: Standardization (for numeric columns)
number_scaler = StandardScaler()
scaled_numbers = pd.DataFrame(
    number_scaler.fit_transform(data[number_columns]),
    columns=number_columns
).round(2)

# Step 9: Combine all processed data
final_data = pd.concat([
    scaled_numbers,    # Standardized numeric columns
    binary_data,       # Binary encoded columns
    location_data,     # One-hot encoded location columns
    provider_data,     # Label encoded provider columns
    count_data         # Frequency encoded description and code columns
], axis=1)

# Step 10: Display final processed data
print("\nFirst 5 rows of processed data:")
print(final_data.head())

# Step 11: Plot the distribution of scaled numeric data
plt.figure(figsize=(12, 6))
scaled_numbers.hist(bins=5, alpha=0.7)
plt.suptitle('Distribution of Scaled Numeric Data')
plt.tight_layout()
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()

# Save the processed data to a new CSV file
final_data.to_csv('iqr_encoding.csv', index=False)
