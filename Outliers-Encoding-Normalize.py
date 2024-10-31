import pandas as pd 
import numpy as np 

df = pd.read_csv('Cleaned_HealthCare.csv')

cols = ['Number of Services', 'Number of Medicare Beneficiaries', 
        'Number of Distinct Medicare Beneficiary/Per Day Services', 
        'Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 
        'Average Medicare Payment Amount', 'Average Medicare Standardized Amount']

#Box Plot to show outliers in the dataset
fig, axs = plt.subplots(nrows=1, ncols=len(cols), figsize=(20, 6))

for i, col in enumerate(cols):
    axs[i].boxplot(df[col])  
    axs[i].set_title(col)  
    axs[i].set_xticks([])  

plt.tight_layout()  
plt.show()

#IQR Method to detect outliers in the dataset
df_iqr = df.copy()
for col in cols:
    Q1 = df_iqr[col].quantile(0.25)
    Q3 = df_iqr[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5*IQR
    upper_bound = Q3 + 1.5*IQR

    df_iqr = df_iqr[(df_iqr[col] >= lower_bound) & (df_iqr[col] <= upper_bound)]

#Box Plot to show outliers after IQR method
fig, axs = plt.subplots(nrows=1, ncols=len(cols), figsize=(20, 6))

for i, col in enumerate(cols):
    axs[i].boxplot(df_iqr[col])  
    axs[i].set_title(col)

plt.tight_layout() 
plt.show()

#Z Score Method to detect outliers in the dataset
df_zscore = df.copy()
for col in cols:
    mean = df_zscore[col].mean()
    std = df_zscore[col].std()
    
    df_zscore['z_score'] = (df_zscore[col] - mean) / std
    
    df_zscore = df_zscore[(df_zscore['z_score'].abs() <= 3)]

df_zscore = df_zscore.drop(columns=['z_score'])

#Box Plot to show outliers after Z Score method
fig, axs = plt.subplots(nrows=1, ncols=len(cols), figsize=(20, 6))

for i, col in enumerate(cols):
    axs[i].boxplot(df_zscore[col])  
    axs[i].set_title(col)

plt.tight_layout() 
plt.show()

cols_c=  ['Gender of the Provider','Entity Type of the Provider','HCPCS Drug Indicator','Medicare Participation Indicator']

#Changing values to T and F
df_encode = pd.get_dummies(df_iqr, columns= cols_c, drop_first= True)
df_encode = df_encode.replace({True: 'T', False: 'F'})
df_encode.head()

df_copy = df_encode.copy()

#Encoding the City of the Providers
city_target_mean = df_copy.groupby('City of the Provider')['Average Medicare Standardized Amount'].mean()
df_copy['City of the Provider Encoded'] = df_copy['City of the Provider'].map(city_target_mean)
overall_mean = df_copy['Average Medicare Standardized Amount'].mean()
df_copy['City of the Provider Encoded'] = df_copy['City of the Provider Encoded'].fillna(overall_mean)
df_copy['City of the Provider Encoded'] = df_copy['City of the Provider Encoded'].round(4)
df_copy[['City of the Provider', 'City of the Provider Encoded']].head(20)

#Encoding the Credentials of the Provider
Credentials_encoded = df_copy['Credentials of the Provider'].value_counts()/len(df_copy)
df_copy['Credentials of the Provider Encoded'] = df_copy['Credentials of the Provider'].map(Credentials_encoded)
df_copy['Credentials of the Provider Encoded'] = df_copy['Credentials of the Provider Encoded'].round(4)
df_copy[['Credentials of the Provider', 'Credentials of the Provider Encoded']].head(20)

#Encoding HCPCS Code
target_mean = df_copy.groupby('HCPCS Code')['Average Medicare Standardized Amount'].mean()
df_copy['HCPCS Code Encoded'] = df_copy['HCPCS Code'].map(target_mean)
overall_mean = df_copy['Average Medicare Standardized Amount'].mean()
df_copy['HCPCS Code Encoded'] = df_copy['HCPCS Code Encoded'].fillna(overall_mean)
df_copy['HCPCS Code Encoded'] = df_copy['HCPCS Code Encoded'].round(4)
df_copy[['HCPCS Code', 'HCPCS Code Encoded']].head(20)

#Normalization of numeric data
cols = ['Number of Services', 'Number of Medicare Beneficiaries', 
        'Number of Distinct Medicare Beneficiary/Per Day Services', 
        'Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 
        'Average Medicare Payment Amount', 'Average Medicare Standardized Amount']
df_copy[cols] = (df_copy[cols] - df_copy[cols].min())/(df_copy[cols].max() - df_copy[cols].min())
df_copy[cols] = df_copy[cols].round(4)

df_copy.to_csv('Updated_HealthCare.csv', index= False)
