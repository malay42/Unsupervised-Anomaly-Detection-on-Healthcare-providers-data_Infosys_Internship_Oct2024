import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings. filterwarnings('ignore')

df = pd.read_csv('Healthcare Providers.csv')
print(df)

print(df.columns)
print(df.info())
print(df.isnull().sum())


DropCols = ['index', 'National Provider Identifier',
       'Last Name/Organization Name of the Provider',
       'First Name of the Provider', 'Middle Initial of the Provider','Street Address 1 of the Provider',
       'Street Address 2 of the Provider','Zip Code of the Provider',"HCPCS Code"]

df = df.drop(DropCols, axis = 1)

print(df.isnull().sum())

print(df.head())

print(df["Entity Type of the Provider"].value_counts())


def RemoveComma(x):
    return x.replace(",","")

df["Average Medicare Allowed Amount"] = pd.to_numeric(df["Average Medicare Allowed Amount"].apply(lambda x: RemoveComma(x)),
                                                             errors= "ignore")
df["Average Submitted Charge Amount"] = pd.to_numeric(df["Average Submitted Charge Amount"].apply(lambda x: RemoveComma(x)),
                                                       errors = "ignore")
df["Average Medicare Payment Amount"] = pd.to_numeric(df["Average Medicare Payment Amount"].apply(lambda x: RemoveComma(x)),
                                                       errors = "ignore")
df["Average Medicare Standardized Amount"] = pd.to_numeric(df["Average Medicare Standardized Amount"].apply(lambda x: RemoveComma(x)),
                                                             errors = "ignore")

print(df)

print(df.info())

df["Credentials of the Provider"] = df["Credentials of the Provider"].fillna(df["Credentials of the Provider"].mode()[0])
df["Gender of the Provider"] = df["Gender of the Provider"].fillna(df["Gender of the Provider"].mode()[0])

print(df.isnull().sum())



# Visualization
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Gender of the Provider')
plt.title('Distribution of Provider Gender')
plt.show()

plt.figure(figsize=(10, 6))
df['Entity Type of the Provider'].value_counts().plot(kind='bar', color='orange', edgecolor='black')
plt.title('Count of Providers by Entity Type', fontsize=16)
plt.xlabel('Entity Type', fontsize=14)
plt.ylabel('Count of Providers', fontsize=14)
plt.xticks(rotation=0)
plt.show()


plt.figure(figsize=(10, 6))
df['Entity Type of the Provider'].value_counts().plot(kind='bar', color='orange', edgecolor='black')
plt.title('Count of Providers by Entity Type', fontsize=16)
plt.xlabel('Entity Type', fontsize=14)
plt.ylabel('Count of Providers', fontsize=14)
plt.xticks(rotation=0)
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Average Medicare Allowed Amount', y='Average Medicare Payment Amount')
plt.title('Allowed vs Payment Amounts')
plt.show()

numerical_columns = ['Number of Services', 'Number of Medicare Beneficiaries', 'Number of Distinct Medicare Beneficiary/Per Day Services', 
                     'Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 'Average Medicare Payment Amount', 
                     'Average Medicare Standardized Amount']

# Remove commas and convert columns to numeric
df[numerical_columns] = df[numerical_columns].replace({',': ''}, regex=True).apply(pd.to_numeric)

# Correlation matrix
corr_matrix = df[numerical_columns].corr()

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Columns')
plt.show()

plt.figure(figsize=(14, 10))
df.groupby('Provider Type')['Average Medicare Standardized Amount'].mean().sort_values().plot(kind='bar', color='purple', edgecolor='black')
plt.title('Average Medicare Standardized Amount by Provider Type', fontsize=16)
plt.xlabel('Provider Type', fontsize=14)
plt.ylabel('Average Medicare Standardized Amount', fontsize=14)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


