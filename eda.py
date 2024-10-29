#!/usr/bin/env python
# coding: utf-8

# **Unsupervised Learning Techniques for Outlier Detection in Healthcare Data**
# 

# 
# The aim of this project is to detect outliers in a healthcare provider dataset and evaluate which outlier detection method performs best. Outliers or anomalies in such data can indicate potential errors, fraud, or unusual patterns that merit further investigation.

# In[1]:


import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('Healthcare Providers.csv')
df.head()


# In[3]:


df.tail()


# In[4]:


df.columns


# In[5]:


df.dtypes


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.isnull().sum()


# In[9]:


num_cols = ["Number of Services","Number of Medicare Beneficiaries",
            "Number of Distinct Medicare Beneficiary/Per Day Services",
            "Average Medicare Allowed Amount","Average Submitted Charge Amount",
            "Average Medicare Payment Amount","Average Medicare Standardized Amount"]

def RemoveComma(x):
    return x.replace(",","")
    
for colm in num_cols:
    df[colm] = pd.to_numeric(df[colm].apply(lambda x: RemoveComma(x)))


# In[10]:


df.info()


# In[11]:


df.loc[(df[num_cols] <1000).all(axis=1)][num_cols].hist(bins=100,figsize=(18,10))


# The histograms show that the majority of the data is concentrated on the left side, indicating a right-skewed distribution.
# Right-skewed distributions often indicate the presence of outliers in the higher range.

# In[12]:


df[['City of the Provider','State Code of the Provider','Country Code of the Provider','Entity Type of the Provider','Provider Type', 'Medicare Participation Indicator', 'Place of Service', 'HCPCS Code']].nunique()


# This section explores the relationships between different attributes in the healthcare provider dataset to uncover patterns, correlations, and potential anomalies. By analyzing how features interact, we can better understand which factors influence each other and where outliers or unusual patterns may emerge. 

# In[13]:


df.groupby(["HCPCS Code"])["index"].count().reset_index().groupby(["index"]).count().head(10)


# In[14]:


df.groupby(["HCPCS Code"])["index"].count().hist(bins=1000,figsize=(17,7));
plt.xlim(0, 500)


# In[15]:


df.groupby(["National Provider Identifier"])["HCPCS Code"].nunique().sort_values(ascending=False).iloc[:10]


# In[16]:


df.groupby(["National Provider Identifier"])["HCPCS Code"].nunique().reset_index().groupby(["HCPCS Code"]).count()


# In[17]:


df.groupby(["National Provider Identifier"])["HCPCS Code"].nunique().hist(bins=30,figsize=(17,7));


# In[18]:


print(df.groupby(["Provider Type"])["index"].count().sort_values(ascending=True))


# In[19]:


df.plot.scatter(x='Number of Services', y='Average Medicare Payment Amount', figsize=(10, 6), alpha=0.5, title='Number of Services vs. Average Medicare Payment Amount')


# In[20]:


n = len(num_cols)  
cols = 2  
rows = (n // cols) + (n % cols > 0)  
for i, col in enumerate(num_cols):
    plt.subplot(rows, cols, i + 1)  
    sns.boxplot(x=df[col])  
    
    plt.xlabel(col)  

plt.tight_layout()  
plt.show()    
    


# In[21]:


corrplot = df[num_cols].corr()
corrplot


# In[22]:


sns.heatmap(corrplot, annot=True,
        xticklabels=corrplot.columns,
        yticklabels=corrplot.columns)


# __Trying Out Different Outlier Detection Methods: Z-Score, IQR, and LOF__
# 

# 1. Z-Score: This method is applied to detect outliers by measuring how many standard deviations a data point is from the mean.

# In[23]:


df1 = df.copy()  # For Z-score method
df2 = df.copy()  # For IQR method
df3 = df.copy()  # For LOF method

z_scores = np.abs((df1[num_cols] - df1[num_cols].mean()) / df1[num_cols].std())
outliers_z = (z_scores > 4)

# Find outlier rows
outliers_z_rows = df1[outliers_z.any(axis=1)]
outliers_z_rows


# In[24]:


df1_cleaned = df1[~outliers_z.any(axis=1)]
corrplot1 = df1_cleaned[num_cols].corr()
sns.heatmap(corrplot1, annot=True,
        xticklabels=corrplot1.columns,
        yticklabels=corrplot1.columns)


# In[37]:


df1_cleaned.shape


# In[25]:


n = len(num_cols)  
cols = 2  
rows = (n // cols) + (n % cols > 0)  
for i, col in enumerate(num_cols):
    plt.subplot(rows, cols, i + 1)  
    sns.boxplot(x=df1_cleaned[col])  
    
    plt.xlabel(col)  

plt.tight_layout()  
plt.show()    
    


# 2. IQR (Interquartile Range): This method uses the spread between the first and third quartiles to find outliers without assuming any specific distribution. It is particularly useful for right-skewed or non-normally distributed data.

# In[54]:


Q1 = df2[num_cols].quantile(0.15)
Q3 = df2[num_cols].quantile(0.75)
IQR = Q3 - Q1

outliers_iqr = ((df2[num_cols] < (Q1 - 1.5 * IQR)) | (df2[num_cols] > (Q3 + 1.5 * IQR)))

# Remove outliers from df2
df2_cleaned = df2[~outliers_iqr.any(axis=1)]
corrplot2 = df2_cleaned[num_cols].corr()
sns.heatmap(corrplot2, annot=True,
        xticklabels=corrplot2.columns,
        yticklabels=corrplot2.columns)


# In[55]:


df2_cleaned.shape


# In[56]:


n = len(num_cols)  
cols = 2  
rows = (n // cols) + (n % cols > 0)  
for i, col in enumerate(num_cols):
    plt.subplot(rows, cols, i + 1)  
    sns.boxplot(x=df2_cleaned[col])  
    
    plt.xlabel(col)  

plt.tight_layout()  
plt.show()    
    


# 3. LOF (Local Outlier Factor): This method is designed to identify local anomalies by comparing the local density of a point to that of its neighbors, making it highly effective for detecting multivariate and density-based outliers.

# In[39]:


from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

scaler = StandardScaler()
df3_scaled = scaler.fit_transform(df3[num_cols])

lof = LocalOutlierFactor(n_neighbors=50)  
outliers_lof = lof.fit_predict(df3_scaled)

# Remove LOF outliers (LOF labels -1 are considered outliers)
df3_cleaned = df3[outliers_lof != -1]
df3_cleaned.shape


# In[40]:


corrplot3 = df3_cleaned[num_cols].corr()
sns.heatmap(corrplot3, annot=True,
        xticklabels=corrplot3.columns,
        yticklabels=corrplot3.columns)


# In[41]:


n = len(num_cols)  
cols = 2  
rows = (n // cols) + (n % cols > 0)  
for i, col in enumerate(num_cols):
    plt.subplot(rows, cols, i + 1)  
    sns.boxplot(x=df3_cleaned[col])  
    
    plt.xlabel(col)  

plt.tight_layout()  
plt.show()    


# Based on the comparisons of boxplots and heatmaps before and after outlier removal, the IQR (Interquartile Range) method was found to be the most effective for outlier detection in the healthcare provider dataset. The boxplots showed a significant reduction in extreme outliers, leading to a more compact distribution, while the heatmaps revealed clearer and more consistent correlations between attributes

# In[57]:


original_count = len(df)
zscore_count = len(df1_cleaned)
iqr_count = len(df2_cleaned)
lof_count = len(df3_cleaned)

outliers_removed_zscore = original_count - zscore_count
outliers_removed_iqr = original_count - iqr_count
outliers_removed_lof = original_count - lof_count

methods = ['Z-Score', 'IQR', 'LOF']
outliers_removed = [outliers_removed_zscore, outliers_removed_iqr, outliers_removed_lof]

plt.figure(figsize=(8, 6))
plt.bar(methods, outliers_removed, color=['blue', 'green', 'red'])
plt.title('Number of Outliers Removed by Each Method')
plt.xlabel('Outlier Detection Method')
plt.ylabel('Number of Outliers Removed')
plt.show()


# In[ ]:




