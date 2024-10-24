# %%
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

file_path = 'Cleaned_Healthcare_Providers.csv'
df = pd.read_csv(file_path)

# Step 1: One-Hot Encoding for non-ordinal categorical columns 
# Columns like 'Provider Type', 'Place of Service', 'HCPCS Drug Indicator' (non-ordinal)
one_hot_columns = ['Provider Type', 'Place of Service', 'HCPCS Drug Indicator']
df = pd.get_dummies(df, columns=one_hot_columns, drop_first=True)

#  Step 2: Label Encoding for binary/ordinal categorical columns
# Columns like 'Gender of the Provider', 'Entity Type of the Providear' (binary/ordinal)
label_encoder = LabelEncoder()
label_columns = ['Gender of the Provider', 'Entity Type of the Provider']

for col in label_columns:
    df[col] = label_encoder.fit_transform(df[col])

#Step 3: Min-Max Normalization for numerical columns 

numerical_columns = ['Number of Services', 'Number of Medicare Beneficiaries', 
                     'Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 
                     'Average Medicare Payment Amount']

scaler = MinMaxScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])


output_file = 'encoded_healthcare_providers.csv'
df.to_csv(output_file, index=False)


# %%

print(df.head(10).to_string(index=False))



# %%
# Display the column names and the total number of columns
print(f"Total columns in the transformed dataset: {len(df.columns)}")
print("Column names:")
print(df.columns)

# Display the shape of the dataset
print(f"Dataset shape (rows, columns): {df.shape}")


# %%
import pandas as pd

# Load the dataset
file_path = 'encoded_healthcare_providers.csv'
df = pd.read_csv(file_path)

# Step 1: Define Constants for Columns
# List of numerical columns to round off
NUMERICAL_COLUMNS = ['Number of Services', 'Number of Medicare Beneficiaries', 
                     'Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 
                     'Average Medicare Payment Amount', 'Average Medicare Standardized Amount']

# Decimal precision
DECIMAL_PLACES = 2

# Step 2: Round the numerical columns to two decimal places
df[NUMERICAL_COLUMNS] = df[NUMERICAL_COLUMNS].round(DECIMAL_PLACES)

# Step 3: Save the updated dataset 
output_file = 'encoded_healthcare_providers.csv'
df.to_csv(output_file, index=False)



