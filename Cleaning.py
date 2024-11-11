import pandas as pd

# Load the data
file_path = "Healthcare Providers.csv"
df = pd.read_csv(file_path)

# total number of null values
null_values = df.isnull().sum()
print(null_values)
missing_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
print(f"\n\nPercentage of missing values: {missing_percentage:.2f}%")


# List of columns to drop based on analysis
columns_to_drop = [
    'index', 'Zip Code of the Provider',
    'Last Name/Organization Name of the Provider','HCPCS Code',
    'Middle Initial of the Provider', 'Credentials of the Provider','HCPCS Description','Provider Type',
    'City of the Provider', 'State Code of the Provider','Street Address 1 of the Provider','Street Address 2 of the Provider'
]

# Drop the specified columns
df = df.drop(columns=columns_to_drop, errors='ignore')

# Display the remaining columns to verify
print("Remaining columns after dropping:")
print(df.columns)

# Save the modified data
output_file_path = "cleaned.csv"
df.to_csv(output_file_path, index=False)
print(f"Data with specified columns dropped saved to {output_file_path}")












dc = pd.read_csv("cleaned.csv")

# total number of null values
null = dc.isnull().sum()
print(null)
missing = (dc.isnull().sum().sum() / (dc.shape[0] * dc.shape[1])) * 100
print(f"\n\nPercentage of missing values: {missing:.2f}%")

# Impute missing values in 'Gender of the Provider' with the mode (most frequent value)
mode_gender = dc['Gender of the Provider'].mode()[0]  # Get the mode of the column
dc['Gender of the Provider'] = dc['Gender of the Provider'].fillna(mode_gender)

mode_name = dc['First Name of the Provider'].mode()[0]  # Get the mode of the column
dc['First Name of the Provider'] = dc['First Name of the Provider'].fillna(mode_name)

# Display to confirm imputation
print("Missing values after mode imputation in 'Gender of the Provider' column:")
print(dc['Gender of the Provider'].isnull().sum())
print(f"Imputed mode value: {mode_gender}")

print("Missing values after mode imputation in 'First Name of the Provider' column:")
print(dc['First Name of the Provider'].isnull().sum())
print(f"Imputed mode value: {mode_name}")


# Save the updated data
output_file_path = "cleaned.csv"
dc.to_csv(output_file_path, index=False)
print(f"Data with mode imputation saved to {output_file_path}")