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
