# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the data
file_path = "C:/Users/shiva/Desktop/PROJECT 1/Healthcare Providers.csv"
df = pd.read_csv(file_path)

# Step 1: Drop unnecessary columns
df = df.drop(columns=['Street Address 2 of the Provider', 'Middle Initial of the Provider'])

# Step 2: Handle missing values
df = df.dropna(subset=['First Name of the Provider', 'Credentials of the Provider'], how='all')
df['Gender of the Provider'].fillna(df['Gender of the Provider'].mode()[0], inplace=True)

# Step 3: Remove duplicates
df = df.drop_duplicates()

# Step 4: Handle numeric columns (remove commas and convert to float)
numeric_columns = [
    'Number of Services', 'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services',
    'Average Medicare Allowed Amount', 'Average Submitted Charge Amount',
    'Average Medicare Payment Amount', 'Average Medicare Standardized Amount'
]
df[numeric_columns] = df[numeric_columns].replace({',': ''}, regex=True).astype(float)

# Step 5: Remove outliers using the IQR method
Q1 = df[numeric_columns].quantile(0.25)
Q3 = df[numeric_columns].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[numeric_columns] < (Q1 - 1.5 * IQR)) | 
           (df[numeric_columns] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Step 6: Normalize and Standardize Data
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Step 7: Clean and standardize the 'Gender of the Provider' column
df['Gender of the Provider'] = (
    df['Gender of the Provider']
    .str.upper()
    .str.replace('.', '', regex=False)
    .str.strip()
)

# Step 8: Clean and standardize the 'Credentials of the Provider' column
df['Credentials of the Provider'] = (
    df['Credentials of the Provider']
    .str.upper()
    .str.replace('.', '', regex=False)
    .str.strip()
    .replace(['M.D', 'MD', 'm.d.'], 'MD')  # Standardize all MD formats
)

# Display final cleaned dataset information
print("Missing values after cleaning:")
print(df.isnull().sum())
print(df.info())
print(df.describe())

# Save the cleaned data
output_file_path = "C:/Users/shiva/Desktop/PROJECT 1/Cleaned_Healthcare_Providers.csv"
df.to_csv(output_file_path, index=False)
print(f"Cleaned data saved to {output_file_path}")
