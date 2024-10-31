import pandas as pd
data = pd.read_csv(r'C:\Users\SHAUN\OneDrive\Desktop\infos\Healthcare Providers.csv')
print(data.head())
print(data.shape)
print(data.info())
#checking for null avlues
print(data.isnull().sum())
miss_data = data.isnull().mean() * 100
print(miss_data[miss_data > 0])
#dropping 
data_dropped = data.drop(columns=["Street Address 2 of the Provider", "Middle Initial of the Provider"])
print(data_dropped.head())
#filling
data_dropped["First Name of the Provider"] = data_dropped["First Name of the Provider"].fillna("Unknown")
data_dropped["Gender of the Provider"] = data_dropped["Gender of the Provider"].fillna("Unknown")
data_dropped["Credentials of the Provider"] = data_dropped["Credentials of the Provider"].fillna("Unknown")
#checking for duplicates
print(data_dropped.isnull().sum())
print(data_dropped.duplicated())
print(data_dropped.duplicated().sum())
print(data_dropped.dtypes)
print(data_dropped.isna().sum())
duplicates = data_dropped.columns[data_dropped.columns.duplicated()].tolist()
print(f"Duplicate columns: {duplicates}")
numeric_columns = [
     'Zip Code of the Provider',
    'Number of Services',
    'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services',
    'Average Medicare Allowed Amount',
    'Average Submitted Charge Amount',
    'Average Medicare Payment Amount',
    'Average Medicare Standardized Amount'
]
#converting into the required type
for column in numeric_columns:
    data_dropped[column] = pd.to_numeric(data_dropped[column] , errors = 'coerce')

print(data_dropped.dtypes)