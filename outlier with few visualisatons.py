import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


numeric_columns = [
    'Number of Services',
    'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services',
    'Average Medicare Allowed Amount',
    'Average Submitted Charge Amount',
    'Average Medicare Payment Amount',
    'Average Medicare Standardized Amount'
]


for col in numeric_columns:
    data_dropped[col] = pd.to_numeric(data_dropped[col], errors='coerce')

def remove_outliers_iqr(df, column, multiplier=1.5): #IQR
    qQ1 = df[column].quantile(0.25)
    qQ3 = df[column].quantile(0.75)
    IQR = qQ3 - qQ1
    lower_bound = qQ1 - (multiplier * IQR)
    upper_bound = qQ3 + (multiplier * IQR)
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


for col in numeric_columns:
    data_dropped = remove_outliers_iqr(data_dropped, col, multiplier=1.2)


plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(data=data_dropped, x=col)
    plt.title(f"Boxplot of {col} (After)")
    plt.ylabel('Value')

plt.tight_layout()
plt.savefig('vis6.png')  # Save the figure as a PNG file

plt.show()


data_test = data_dropped.copy()

from scipy import stats
import numpy as np


relevant_columns = [
    'Number of Services', 
    'Number of Medicare Beneficiaries', 
    'Number of Distinct Medicare Beneficiary/Per Day Services', 
    'Average Medicare Allowed Amount', 
    'Average Submitted Charge Amount', 
    'Average Medicare Payment Amount'
]


def replace_outliers_with_median_zscore(df, columns, threshold=3): #Z score
    df_clean = df.copy()
    for col in columns:
        
        z_scores = np.abs(stats.zscore(df_clean[col]))
        
        
        median_value = df_clean[col].median()
        
        
        df_clean[col] = np.where(z_scores > threshold, median_value, df_clean[col])
    
    return df_clean


data_zscore_replaced = replace_outliers_with_median_zscore(data_test, relevant_columns)


plt.figure(figsize=(15, 10))

for i, col in enumerate(relevant_columns, 1):
    plt.subplot(3, 2, i)
    sns.boxplot(data_zscore_replaced[col], color="blue")
    plt.title(f"Boxplot of {col} (Outliers replaced using Z-score)")
    plt.tight_layout()

plt.savefig('vis7.png')  # Save the figure as a PNG file
data_test1 = data_dropped.copy()
data_test1.to_csv('data_test1.csv', index=False)
loaded_data = pd.read_csv('data_test1.csv')

missing_values = data_test1.isnull().sum()


print(missing_values[missing_values > 0])


total_missing_values = data_test1.isnull().sum().sum()
print(f'Total missing values in the dataset: {total_missing_values}')
data_test1[relevant_columns] = data_test1[relevant_columns].replace(',', '', regex=True).apply(pd.to_numeric)
import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor


relevant_columns = [
    'Number of Services', 
    'Number of Medicare Beneficiaries', 
    'Number of Distinct Medicare Beneficiary/Per Day Services', 
    'Average Medicare Allowed Amount', 
    'Average Submitted Charge Amount', 
    'Average Medicare Payment Amount'
]


#LOCAL OUTLIER FACTOR
data_for_lof = data_test1[relevant_columns]
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)  
outliers = lof.fit_predict(data_for_lof)
data_test1['LOF_Outlier'] = outliers  
outlier_data = data_test1[data_test1['LOF_Outlier'] == -1]  
print("Detected Outliers:")
print(outlier_data)
outlier_data.to_csv('detected_outliers.csv', index=False)
data_cleaned_lof = data_test1[data_test1['LOF_Outlier'] == 1].copy()  
data_cleaned_lof.drop('LOF_Outlier', axis=1, inplace=True)
data_test1[relevant_columns] = data_test1[relevant_columns].replace(',', '', regex=True).apply(pd.to_numeric, errors='coerce')
plt.figure(figsize=(15, 10))
for i, column in enumerate(relevant_columns, start=1):
    plt.subplot(2, 3, i)  
    sns.boxplot(y=data_test1[column])
    plt.title(column)
    plt.ylim(0, data_test1[column].max() + 100)  

plt.tight_layout()

plt.savefig('vis8.png') 