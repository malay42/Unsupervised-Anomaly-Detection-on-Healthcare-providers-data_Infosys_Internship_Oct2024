#importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import missingno as msno
#reading and loading the dataset
df=pd.read_csv('Healthcare Providers.csv')
#basic informations 
#number of rows and cols
print(df.shape)
# printing number of non null values and datatype of columns
print(df.info())
#Summary Statistics
print(df.describe())
#Missing values per column
df.isnull().sum()
#visualising the missing values in the dataframe using bargraph 
msno.bar(df,color='blue',sort='ascending',figsize=(10,5),fontsize=15)
# visualising the missing values in the dataframe using heatmap
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.figure(figsize=(10,5))
sns.heatmap(df.isna(),cmap='coolwarm')
# Define the threshold for missing values (20% of the total number of columns)
threshold = int(df.shape[1] * 0.2)

# Drop rows with more than 20% missing values
df = df.dropna(thresh=df.shape[1] - threshold)

print("Shape after dropping rows with >20% missing values:", df.shape)
# Fill missing values in 'First Name of the Providers' with values from 'Middle Name'

df['First Name of the Provider'] = df['First Name of the Provider'].fillna(df['Middle Initial of the Provider'])
# Drop the 'Middle Name' column
df.drop(columns=['Middle Initial of the Provider'], inplace=True)

# Check for missing values in the 'Last Name' column
missing_last_name = df['Last Name/Organization Name of the Provider'].isnull().sum()

if missing_last_name > 0:
    print(f"There are {missing_last_name} missing values in the 'Last Name' column.")
else:
    print("There are no missing values in the 'Last Name' column.")
# Fill missing values in 'First Name of the Providers' with values from 'last Name'

df['First Name of the Provider'] = df['First Name of the Provider'].fillna(df['Last Name/Organization Name of the Provider'])
#drop street 2 
df.drop(columns=['Street Address 2 of the Provider'], inplace=True)
# check the columns having missing values
df.isna().sum()
#printing the number of rows and columns to check whether the columns are removed or not
print(df.shape)
# Fill missing values in numerical columns with the median
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col] = df[col].fillna(df[col].median())

# Fill missing values in categorical columns with the mode
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])
    #removing the duplicate values
df.drop_duplicates(inplace=True)
# check if there is missing values or not
df.isna().sum()
#printing the the dataset
df
#visualising the missing values in the dataframe using bargraph after removing them
msno.bar(df,color='blue',sort='ascending',figsize=(10,5),fontsize=15)
#visualizing missing values using heatmap after removing them
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.figure(figsize=(10,5))
sns.heatmap(df.isna(),cmap='coolwarm')

# Save the cleaned data to a new CSV file
df.to_csv('cleaned_data.csv', index=False)
print("\nData cleaning complete. Cleaned data saved as 'cleaned_data.csv'.")