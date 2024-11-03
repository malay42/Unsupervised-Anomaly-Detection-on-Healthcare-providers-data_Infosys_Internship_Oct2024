import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler


df1=pd.read_csv('Cleaned_Healthcare Providers.csv')
print(df1)
numerical_columns = ['Number of Services', 'Number of Medicare Beneficiaries', 
                     'Number of Distinct Medicare Beneficiary/Per Day Services',
                     'Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 
                     'Average Medicare Payment Amount', 'Average Medicare Standardized Amount']

# Create a copy of the dataset for IQR-based outlier removal
df11 = df1.copy()  # Assuming df1 is your original dataset

#IQR Method to remove outliers from the numerical columns
def remove_outliers_iqr(df11, numerical_columns):
    # Calculate Q1 (25th percentile) and Q3 (75th percentile) for numerical columns
    Q1 = df11[numerical_columns].quantile(0.25)
    Q3 = df11[numerical_columns].quantile(0.75)
    IQR = Q3 - Q1  # Calculate the Interquartile Range (IQR)
    
    # Remove rows with outliers
    return df11[~((df11[numerical_columns] < (Q1 - 1.5 * IQR)) | (df11[numerical_columns] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Apply the IQR method to remove outliers
df11 = remove_outliers_iqr(df11, numerical_columns)

print(f"Original Data Shape: {df1.shape}, New Data Shape after IQR Outlier Removal: {df11.shape}")




#Z_Score Method To Remove Outliers
df10=df1.copy()
def remove_outliers_zscore(df10, threshold=3):
    z_scores = np.abs((df10 - df10.mean()) / df10.std())
    return df10[(z_scores < threshold).all(axis=1)]

df10 = remove_outliers_zscore(df10[numerical_columns])
print(f"Original Data Shape: {df1.shape}, New Data Shape after Z-Score Outlier Removal: {df10.shape}")



# Boxplot before removing outliers
plt.figure(figsize=(15, 8))
sns.boxplot(data=df1[numerical_columns])
plt.title('Boxplot of Numerical Features Before Outlier Removal')
plt.xticks(rotation=45)
plt.show()

# Boxplot after removing outliers using Z-Score
plt.figure(figsize=(15, 8))
sns.boxplot(data=df10)
plt.title('Boxplot of Numerical Features After Z-Score Outlier Removal')
plt.xticks(rotation=45)
plt.show()

# Boxplot after removing outliers using IQR
plt.figure(figsize=(15, 8))
sns.boxplot(data=df11)
plt.title('Boxplot of Numerical Features After IQR Outlier Removal')
plt.xticks(rotation=45)
plt.show()


#One-Hot Encoding
# List of categorical columns for One-Hot Encoding
one_hot_columns = ['Credentials of the Provider', 'Entity Type of the Provider', 
                   'Place of Service', 'Country Code of the Provider', 
                   'Medicare Participation Indicator', 'HCPCS Drug Indicator', 
                   'Gender of the Provider']

# Ensure that these columns exist in the dataset
available_one_hot_columns = [col for col in one_hot_columns if col in df11.columns]

# Apply One-Hot Encoding and drop the first category to avoid multicollinearity (using drop_first=True)
df11 = pd.get_dummies(df11, columns=available_one_hot_columns, drop_first=True)

# Step 4: Frequency Encoding
# List of columns to apply Frequency Encoding
columns_to_encode = ['City of the Provider', 'State Code of the Provider', 'Provider Type', 'HCPCS Code']

# Frequency Encoding for specified columns
for column in columns_to_encode:
    if column in df1.columns:  # Ensure the column exists in the original DataFrame
        frequency_counts = df1[column].value_counts()  # Calculate frequency counts based on the original dataset
        df11[column] = df11[column].map(frequency_counts)  # Map the frequency counts to df11

# Step 5: Print the shape of the final dataset
print(f"Final Data Shape after encoding: {df11.shape}")


# Print the final DataFrame after conversion
print(df11.head())

boolean_columns = df11.select_dtypes(include='bool').columns

# Convert boolean columns to integers (0 and 1)
df11[boolean_columns] = df11[boolean_columns].astype(int)

# Normalization using MinMaxScaler
min_max_scaler = MinMaxScaler()
df11[numerical_columns] = min_max_scaler.fit_transform(df11[numerical_columns])

# Print DataFrame after normalization
print("DataFrame after normalization:")
print(df11.head())

# Standardization using StandardScaler
standard_scaler = StandardScaler()
df11[numerical_columns] = standard_scaler.fit_transform(df11[numerical_columns])

# Print DataFrame after standardization
print("\nDataFrame after standardization:")
print(df11.head())

#df11.to_csv('Transformed_Data.csv', index= False)