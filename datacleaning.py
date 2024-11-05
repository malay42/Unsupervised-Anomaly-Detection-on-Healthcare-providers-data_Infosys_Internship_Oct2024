import numpy as np
import seaborn as sns
import matplotlib as plt
df=pd.read_csv('Healthcare Providers.csv')
df.isnull().sum()
import missingno as msno
msno.bar(df,color='blue',sort='ascending',figsize=(10,5),fontsize=15)
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.figure(figsize=(10,5))
sns.heatmap(df.isna(),cmap='coolwarm')
df.dropna(inplace=True)
df.isna().sum()
df.fillna(value=0,inplace=True)
df.drop_duplicates(inplace=True)
df
