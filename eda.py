
# Importing Libraries
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

# Loading the Data
df = pd.read_csv('Healthcare Providers.csv')
df.head()

# Basic Data Exploration
df.tail()
df.columns
df.dtypes
df.info()
df.describe()
df.isnull().sum()

# Data Preprocessing: Converting Numeric Columns
num_cols = ["Number of Services", "Number of Medicare Beneficiaries",
            "Number of Distinct Medicare Beneficiary/Per Day Services",
            "Average Medicare Allowed Amount", "Average Submitted Charge Amount",
            "Average Medicare Payment Amount", "Average Medicare Standardized Amount"]

def RemoveComma(x):
    return x.replace(",", "")

for colm in num_cols:
    df[colm] = pd.to_numeric(df[colm].apply(lambda x: RemoveComma(x)))

df.info()

# Visualizing Data Distributions
df.loc[(df[num_cols] < 1000).all(axis=1)][num_cols].hist(bins=100, figsize=(18, 10))

# Unique Values in Categorical Columns
df[['City of the Provider', 'State Code of the Provider', 'Country Code of the Provider',
    'Entity Type of the Provider', 'Provider Type', 'Medicare Participation Indicator', 
    'Place of Service', 'HCPCS Code']].nunique()

# Analyzing HCPCS Code Frequency
df.groupby(["HCPCS Code"])["index"].count().reset_index().groupby(["index"]).count().head(10)
df.groupby(["HCPCS Code"])["index"].count().hist(bins=1000, figsize=(17, 7))
plt.xlim(0, 500)

# Analyzing National Provider Identifier and HCPCS Code Count
df.groupby(["National Provider Identifier"])["HCPCS Code"].nunique().sort_values(ascending=False).iloc[:10]
df.groupby(["National Provider Identifier"])["HCPCS Code"].nunique().reset_index().groupby(["HCPCS Code"]).count()
df.groupby(["National Provider Identifier"])["HCPCS Code"].nunique().hist(bins=30, figsize=(17, 7))

# Provider Type Distribution
print(df.groupby(["Provider Type"])["index"].count().sort_values(ascending=True))

# Scatter Plot: Services vs Medicare Payment Amount
df.plot.scatter(x='Number of Services', y='Average Medicare Payment Amount', figsize=(10, 6), alpha=0.5, title='Number of Services vs. Average Medicare Payment Amount')

# Boxplots for Numerical Columns
n = len(num_cols)
cols = 2
rows = (n // cols) + (n % cols > 0)
for i, col in enumerate(num_cols):
    plt.subplot(rows, cols, i + 1)
    sns.boxplot(x=df[col])
    plt.xlabel(col)
plt.tight_layout()
plt.show()

# Correlation Heatmap
corrplot = df[num_cols].corr()
sns.heatmap(corrplot, annot=True, xticklabels=corrplot.columns, yticklabels=corrplot.columns)

# Outlier Detection Methods: Z-Score
df1 = df.copy()
z_scores = np.abs((df1[num_cols] - df1[num_cols].mean()) / df1[num_cols].std())
outliers_z = (z_scores > 4)
outliers_z_rows = df1[outliers_z.any(axis=1)]
df1_cleaned = df1[~outliers_z.any(axis=1)]
corrplot1 = df1_cleaned[num_cols].corr()
sns.heatmap(corrplot1, annot=True, xticklabels=corrplot1.columns, yticklabels=corrplot1.columns)

# Z-Score Boxplots After Cleaning
df1_cleaned.shape
n = len(num_cols)
cols = 2
rows = (n // cols) + (n % cols > 0)
for i, col in enumerate(num_cols):
    plt.subplot(rows, cols, i + 1)
    sns.boxplot(x=df1_cleaned[col])
    plt.xlabel(col)
plt.tight_layout()
plt.show()

# Outlier Detection Methods: IQR
df2 = df.copy()
Q1 = df2[num_cols].quantile(0.15)
Q3 = df2[num_cols].quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = ((df2[num_cols] < (Q1 - 1.5 * IQR)) | (df2[num_cols] > (Q3 + 1.5 * IQR)))
df2_cleaned = df2[~outliers_iqr.any(axis=1)]
corrplot2 = df2_cleaned[num_cols].corr()
sns.heatmap(corrplot2, annot=True, xticklabels=corrplot2.columns, yticklabels=corrplot2.columns)

# IQR Boxplots After Cleaning
df2_cleaned.shape
n = len(num_cols)
cols = 2
rows = (n // cols) + (n % cols > 0)
for i, col in enumerate(num_cols):
    plt.subplot(rows, cols, i + 1)
    sns.boxplot(x=df2_cleaned[col])
    plt.xlabel(col)
plt.tight_layout()
plt.show()

# Outlier Detection Methods: LOF (Local Outlier Factor)
scaler = StandardScaler()
df3_scaled = scaler.fit_transform(df3[num_cols])
lof = LocalOutlierFactor(n_neighbors=50)
outliers_lof = lof.fit_predict(df3_scaled)
df3_cleaned = df3[outliers_lof != -1]
df3_cleaned.shape
corrplot3 = df3_cleaned[num_cols].corr()
sns.heatmap(corrplot3, annot=True, xticklabels=corrplot3.columns, yticklabels=corrplot3.columns)

# LOF Boxplots After Cleaning
n = len(num_cols)
cols = 2
rows = (n // cols) + (n % cols > 0)
for i, col in enumerate(num_cols):
    plt.subplot(rows, cols, i + 1)
    sns.boxplot(x=df3_cleaned[col])
    plt.xlabel(col)
plt.tight_layout()
plt.show()

# Comparison of Outlier Removal
original_count = len(df)
zscore_count = len(df1_cleaned)
iqr_count = len(df2_cleaned)
lof_count = len(df3_cleaned)

outliers_removed_zscore = original_count - zscore_count
outliers_removed_iqr = original_count - iqr_count
outliers_removed_lof = original_count - lof_count

methods = ['Z-Score', 'IQR', 'LOF']
outliers_removed = [outliers_removed_zscore, outliers_removed_iqr, outliers_removed_lof]

# Plotting Outliers Removed by Each Method
plt.figure(figsize=(8, 6))
plt.bar(methods, outliers_removed, color=['blue', 'green', 'red'])
plt.title('Number of Outliers Removed by Each Method')
plt.xlabel('Outlier Detection Method')
plt.ylabel('Number of Outliers Removed')
plt.show()
