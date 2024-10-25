import pandas as pd
df=pd.read_csv('Healthcare Providers.csv')

df = df.drop(columns=['Street Address 2 of the Provider','Middle Initial of the Provider'])
df = df.dropna(subset=['First Name of the Provider'], how='all')
most_frequent_gender = df['Gender of the Provider'].mode()[0]
df['Gender of the Provider'] = df['Gender of the Provider'].fillna(most_frequent_gender)
df = df.dropna(subset=['Credentials of the Provider'], how='all')
df = df.drop_duplicates()
df.isnull().sum()