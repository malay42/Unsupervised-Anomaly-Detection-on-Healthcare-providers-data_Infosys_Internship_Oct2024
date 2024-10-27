import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Load the dataset
df = pd.read_csv("C:\\Users\\malla\\Downloads\\archive\\Healthcare Providers.csv")

# Specify numeric columns to analyze
numeric_columns = [
'Number of Services',
'Number of Medicare Beneficiaries',
'Number of Distinct Medicare Beneficiary/Per Day Services',
'Average Medicare Allowed Amount',
'Average Submitted Charge Amount',
'Average Medicare Payment Amount',
'Average Medicare Standardized Amount'
]

# Method 1: Isolation Forest
def isolation_forest_outliers(df, contamination=0.05):
iso_forest = IsolationForest(contamination=contamination, random_state=42)
df['Anomaly_IF'] = iso_forest.fit_predict(df[numeric_columns])
return df[df['Anomaly_IF'] == 1].drop(columns=['Anomaly_IF'])

# Method 2: Local Outlier Factor
def local_outlier_factor(df, contamination=0.05):
lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
df['Anomaly_LOF'] = lof.fit_predict(df[numeric_columns])
return df[df['Anomaly_LOF'] == 1].drop(columns=['Anomaly_LOF'])

# Method 3: Hybrid Approach using Z-Score and IQR
def hybrid_outliers(df, col, z_thresh=3, iqr_multiplier=1.5):
# Z-Score filtering
    z_filtered = df[np.abs((df[col] - df[col].mean()) / df[col].std()) < z_thresh]

# IQR filtering
Q1 = z_filtered[col].quantile(0.25)
Q3 = z_filtered[col].quantile(0.75)
IQR = Q3 - Q1
iqr_filtered = z_filtered[(z_filtered[col] >= (Q1 - iqr_multiplier * IQR)) &
                               (z_filtered[col] <= (Q3 + iqr_multiplier * IQR))]
return iqr_filtered

# Apply Isolation Forest
df_if_cleaned = isolation_forest_outliers(df)

# Apply Local Outlier Factor
df_lof_cleaned = local_outlier_factor(df)

# Apply Hybrid Method to each numeric column
df_hybrid_cleaned = df.copy()
for col in numeric_columns:
df_hybrid_cleaned = hybrid_outliers(df_hybrid_cleaned, col)

# Compare the cleaned dataframes
print("Rows after Isolation Forest:", df_if_cleaned.shape[0])
print("Rows after Local Outlier Factor:", df_lof_cleaned.shape[0])
print("Rows after Hybrid Approach:", df_hybrid_cleaned.shape[0])

# Visualize the results (optional)
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numeric_columns])
plt.title('Box Plot of Numeric Columns Before Outlier Removal')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df_hybrid_cleaned[numeric_columns])
plt.title('Box Plot of Numeric Columns After Hybrid Outlier Removal')
plt.show()
