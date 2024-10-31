import pandas as pd
data = pd.read_csv(r'C:\Users\SHAUN\OneDrive\Desktop\infos\Healthcare Providers.csv')
print(data.head())
print(data.shape)
print(data.info())
print(data.isnull().sum())
miss_data = data.isnull().mean() * 100
print(miss_data[miss_data > 0])
data_dropped = data.drop(columns=["Street Address 2 of the Provider", "Middle Initial of the Provider"])
print(data_dropped.head())
data_dropped["First Name of the Provider"] = data_dropped["First Name of the Provider"].fillna("Unknown")
data_dropped["Gender of the Provider"] = data_dropped["Gender of the Provider"].fillna("Unknown")
data_dropped["Credentials of the Provider"] = data_dropped["Credentials of the Provider"].fillna("Unknown")
print(data_dropped.isnull().sum())
print(data_dropped.duplicated())
print(data_dropped.duplicated().sum())
print(data_dropped.dtypes)
print(data_dropped.isna().sum())
duplicates = data_dropped.columns[data_dropped.columns.duplicated()].tolist()
print(f"Duplicate columns: {duplicates}")
numeric_columns = [
     'Zip Code of the Provider',
    'Number of Services',
    'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services',
    'Average Medicare Allowed Amount',
    'Average Submitted Charge Amount',
    'Average Medicare Payment Amount',
    'Average Medicare Standardized Amount'
]
for column in numeric_columns:
    data_dropped[column] = pd.to_numeric(data_dropped[column] , errors = 'coerce')

print(data_dropped.dtypes)
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend

import matplotlib.pyplot as plt
import seaborn as sns


# Define the size of the figure and number of subplots
plt.figure(figsize=(15, 12))

# List of column names for iteration
columns = [
    'Number of Services',
    'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services',
    'Average Medicare Allowed Amount',
    'Average Submitted Charge Amount',
    'Average Medicare Payment Amount',
    'Average Medicare Standardized Amount'
]

# Loop through each column to create subplots
for i, col in enumerate(columns, 1):
    plt.subplot(3, 3, i)  # 3 rows, 3 columns
    sns.histplot(data_dropped[col], kde=True)  # Using histplot for better visual representation with kde curve
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

# Adjust layout
plt.tight_layout()

# Display all the subplots
plt.savefig('output_figure.png')  # Save the figure as a PNG file

summary_stats = data_dropped.describe()
print("Summary Statistics:\n", summary_stats)


plt.figure(figsize=(18, 12))


plt.subplot(3, 3, 1)
sns.histplot(data=data_dropped, x='Number of Services', kde=False, bins=30)
plt.title('Distribution of Number of Services')


plt.subplot(3, 3, 2)
sns.histplot(data=data_dropped, x='Number of Medicare Beneficiaries', kde=False, bins=30)
plt.title('Distribution of Number of Medicare Beneficiaries')


plt.subplot(3, 3, 3)
sns.histplot(data=data_dropped, x='Number of Distinct Medicare Beneficiary/Per Day Services', kde=False, bins=30)
plt.title('Distribution of Distinct Medicare Beneficiary/Per Day Services')


plt.subplot(3, 3, 4)
sns.histplot(data=data_dropped, x='Average Medicare Allowed Amount', kde=False, bins=30)
plt.title('Distribution of Average Medicare Allowed Amount')


plt.subplot(3, 3, 5)
sns.histplot(data=data_dropped, x='Average Submitted Charge Amount', kde=False, bins=30)
plt.title('Distribution of Average Submitted Charge Amount')


plt.subplot(3, 3, 6)
sns.histplot(data=data_dropped, x='Average Medicare Payment Amount', kde=False, bins=30)
plt.title('Distribution of Average Medicare Payment Amount')


plt.subplot(3, 3, 7)
sns.histplot(data=data_dropped, x='Average Medicare Standardized Amount', kde=False, bins=30)
plt.title('Distribution of Average Medicare Standardized Amount')

plt.tight_layout()
plt.savefig('Visualisation1.png')  # Save the figure as a PNG file

plt.show()


plt.figure(figsize=(18, 12))


plt.subplot(3, 3, 1)
sns.boxplot(x=data_dropped['Number of Services'])
plt.title('Boxplot - Number of Services')


plt.subplot(3, 3, 2)
sns.boxplot(x=data_dropped['Number of Medicare Beneficiaries'])
plt.title('Boxplot - Number of Medicare Beneficiaries')


plt.subplot(3, 3, 3)
sns.boxplot(x=data_dropped['Number of Distinct Medicare Beneficiary/Per Day Services'])
plt.title('Boxplot - Distinct Medicare Beneficiary/Per Day Services')

plt.subplot(3, 3, 4)
sns.boxplot(x=data_dropped['Average Medicare Allowed Amount'])
plt.title('Boxplot - Average Medicare Allowed Amount')


plt.subplot(3, 3, 5)
sns.boxplot(x=data_dropped['Average Submitted Charge Amount'])
plt.title('Boxplot - Average Submitted Charge Amount')


plt.subplot(3, 3, 6)
sns.boxplot(x=data_dropped['Average Medicare Payment Amount'])
plt.title('Boxplot - Average Medicare Payment Amount')


plt.subplot(3, 3, 7)
sns.boxplot(x=data_dropped['Average Medicare Standardized Amount'])
plt.title('Boxplot - Average Medicare Standardized Amount')

plt.tight_layout()
plt.savefig('Visualisation2.png')  # Save the figure as a PNG file
columns_of_interest = [
    'Number of Services',
    'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services',
    'Average Medicare Allowed Amount',
    'Average Submitted Charge Amount',
    'Average Medicare Payment Amount',
    'Average Medicare Standardized Amount'
]


correlation_matrix = data_dropped[columns_of_interest].corr()


plt.figure(figsize=(10, 6))


sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})


plt.title('Correlation Heatmap of Selected Features')


plt.savefig('Visualisation3.png')  # Save the figure as a PNG file
plt.figure(figsize=(10, 6))
plt.scatter(data_dropped['Number of Medicare Beneficiaries'], data_dropped['Average Medicare Payment Amount'], alpha=0.5)
plt.title('Scatter plot of Number of Medicare Beneficiaries vs Average Medicare Payment Amount')
plt.xlabel('Number of Medicare Beneficiaries')
plt.ylabel('Average Medicare Payment Amount')
plt.grid(True)
plt.savefig('vis4.png')  # Save the figure as a PNG file
import pandas as pd
import matplotlib.pyplot as plt

# Define the categorical columns
categorical_columns = [
    'Gender of the Provider',
    'Entity Type of the Provider',
    'Medicare Participation Indicator',
    'HCPCS Drug Indicator'
]

# Loop through each categorical column to generate a pie chart
for col in categorical_columns:
    # Count the occurrences of each category in the column
    category_counts = data_dropped[col].value_counts()

    # Create the pie chart
    plt.figure(figsize=(7, 7))
    plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
    plt.title(f'Distribution of {col}')
    plt.show()
plt.savefig('vis5.png')  # Save the figure as a PNG file
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

def remove_outliers_iqr(df, column, multiplier=1.5):
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


def replace_outliers_with_median_zscore(df, columns, threshold=3):
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

plt.savefig('vis8.png')  # Save the figure as a PNG file
from sklearn.preprocessing import LabelEncoder
import pandas as pd

label_encoding_columns = [
    'Gender of the Provider',
    'Entity Type of the Provider',
    'Medicare Participation Indicator',
    'HCPCS Drug Indicator',
    'Credentials of the Provider',  
    'HCPCS Code',  
    'HCPCS Description'  
]

one_hot_encoding_columns = [
    'Provider Type',
    'Place of Service',
    'State Code of the Provider',
    'Country Code of the Provider'
]


label_encoder = LabelEncoder()


for col in label_encoding_columns:
    if col in data_dropped.columns:  
        data_dropped[col] = label_encoder.fit_transform(data_dropped[col])


data_dropped = pd.get_dummies(data_dropped, columns=one_hot_encoding_columns, drop_first=False)


data_dropped = data_dropped.replace({True: 'T', False: 'F'})


print(data_dropped.head())
print(data_dropped.isnull().sum())

numerical_columns_count = data_dropped.select_dtypes(include=['int64', 'float64']).shape[1]


print("Number of numerical columns:", numerical_columns_count)
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(style="whitegrid")


numerical_columns = [
    'Average Medicare Payment Amount',
    'Average Medicare Allowed Amount',
    'Average Submitted Charge Amount',
    'Number of Services',
    'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services',
    'Average Medicare Standardized Amount'
]

plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(3, 3, i)
    plt.hist(data_dropped[col], bins=30, edgecolor='black')
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(True)

plt.tight_layout()
plt.show()


categorical_columns = [
    'Gender of the Provider',
    'Entity Type of the Provider',
    'Medicare Participation Indicator',
    'HCPCS Drug Indicator'
]

plt.figure(figsize=(15, 10))
for i, col in enumerate(categorical_columns, 1):
    plt.subplot(2, 2, i)
    sns.countplot(data=data_dropped, x=col)
    plt.title(f'Count of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
plt.scatter(data_dropped['Number of Medicare Beneficiaries'], data_dropped['Average Medicare Payment Amount'], alpha=0.5)
plt.title('Scatter plot of Number of Medicare Beneficiaries vs Average Medicare Payment Amount')
plt.xlabel('Number of Medicare Beneficiaries')
plt.ylabel('Average Medicare Payment Amount')
plt.grid(True)
plt.show()



plt.figure(figsize=(10, 6))
sns.heatmap(data_dropped[numerical_columns].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Variables')
plt.savefig('vis9.png') 
# Save the figure as a PNG file
# Display only numerical columns
numerical_columns = data_dropped.select_dtypes(include=['number'])

# Show the first few rows of the numerical columns
numerical_columns.head()
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
data = data_dropped
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

normalize_cols = [
    'Number of Services',
    'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services'
]

standardize_cols = [
    'Average Medicare Allowed Amount',
    'Average Submitted Charge Amount',
    'Average Medicare Payment Amount',
    'Average Medicare Standardized Amount'
]

min_max_scaler = MinMaxScaler()
data_dropped[normalize_cols] = min_max_scaler.fit_transform(data_dropped[normalize_cols])
scaler = StandardScaler()
data_dropped[standardize_cols] = scaler.fit_transform(data_dropped[standardize_cols])
print(data_dropped.head())
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

numeric_data = data_dropped.select_dtypes(include=['float64', 'int64'])


iso_forest = IsolationForest(n_estimators=100, contamination=0.08, random_state=42)


iso_forest.fit(numeric_data)


data_dropped['Anomaly'] = iso_forest.predict(numeric_data)


anomalies = data_dropped[data_dropped['Anomaly'] == -1]


plt.figure(figsize=(12, 6))


plt.scatter(data_dropped[data_dropped['Anomaly'] == 1]['Number of Services'],
            data_dropped[data_dropped['Anomaly'] == 1]['Average Medicare Payment Amount'],
            label='Normal', alpha=0.5)

plt.scatter(anomalies['Number of Services'],
            anomalies['Average Medicare Payment Amount'],
            color='red', label='Anomaly', alpha=0.7)
plt.title('Anomaly Detection: Number of Services vs Average Medicare Payment Amount')
plt.xlabel('Number of Services')
plt.ylabel('Average Medicare Payment Amount')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 6))

plt.scatter(data_dropped[data_dropped['Anomaly'] == 1]['Number of Medicare Beneficiaries'],
            data_dropped[data_dropped['Anomaly'] == 1]['Average Medicare Allowed Amount'],
            label='Normal', alpha=0.5)
plt.scatter(anomalies['Number of Medicare Beneficiaries'],
            anomalies['Average Medicare Allowed Amount'],
            color='red', label='Anomaly', alpha=0.7)
plt.title('Anomaly Detection: Number of Medicare Beneficiaries vs Average Medicare Allowed Amount')
plt.xlabel('Number of Medicare Beneficiaries')
plt.ylabel('Average Medicare Allowed Amount')
plt.legend()
plt.grid()
plt.savefig('isoforest.png')  # Save the figure as a PNG file

print("Anomalies detected:")
print(anomalies)
print(f"Total number of anomalies: {anomalies.shape[0]}")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

numeric_data = data_dropped[['Number of Services', 'Number of Medicare Beneficiaries',
                             'Number of Distinct Medicare Beneficiary/Per Day Services',
                             'Average Medicare Allowed Amount', 'Average Submitted Charge Amount',
                             'Average Medicare Payment Amount', 'Average Medicare Standardized Amount']]

dbscan = DBSCAN(eps=0.5, min_samples=5)
data_dropped['DBSCAN_Label'] = dbscan.fit_predict(numeric_data)


anomalies = data_dropped[data_dropped['DBSCAN_Label'] == -1]
normal_data = data_dropped[data_dropped['DBSCAN_Label'] != -1]


plt.figure(figsize=(10, 6))

plt.scatter(normal_data['Number of Services'], normal_data['Average Medicare Payment Amount'],
            label='Normal', alpha=0.5)
plt.scatter(anomalies['Number of Services'], anomalies['Average Medicare Payment Amount'],
            color='red', label='Anomaly', alpha=0.7)


plt.title('DBSCAN Anomaly Detection: Number of Services vs Average Medicare Payment Amount')
plt.xlabel('Number of Services')
plt.ylabel('Average Medicare Payment Amount')
plt.legend()
plt.grid()
plt.savefig('dbscan.png')  

print("Anomalies detected by DBSCAN:")
print(anomalies)
print(f"Total number of anomalies: {anomalies.shape[0]}")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import numpy as np


numeric_data = data_dropped.select_dtypes(include=['float64', 'int64'])

nbrs = NearestNeighbors(n_neighbors=5).fit(numeric_data)
distances, indices = nbrs.kneighbors(numeric_data)

threshold = np.percentile(distances[:, 4], 95)
data_dropped['Anomaly'] = (distances[:, 4] > threshold).astype(int)  
anomalies = data_dropped[data_dropped['Anomaly'] == 1]
print("Anomalies detected by KNN:")
print(anomalies)
print(f"Total number of anomalies: {anomalies.shape[0]}")
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(numeric_data)


plt.figure(figsize=(10, 6))
plt.scatter(reduced_data[data_dropped['Anomaly'] == 0, 0],
            reduced_data[data_dropped['Anomaly'] == 0, 1],
            c='blue', label='Normal', alpha=0.6, edgecolors='w', s=50)


plt.scatter(reduced_data[data_dropped['Anomaly'] == 1, 0],
            reduced_data[data_dropped['Anomaly'] == 1, 1],
            c='red', label='Anomaly', alpha=0.8, edgecolors='w', s=50)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('KNN Anomaly Detection Results')
plt.legend()
plt.savefig('knn.png')  # Save the figure as a PNG file
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt

# Select only numerical columns from data_dropped
numerical_data = data_dropped.select_dtypes(include=[np.number]).values

# Fit Gaussian Mixture Model
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
gmm.fit(numerical_data)

# Get probabilities for each point
probabilities = gmm.score_samples(numerical_data)

# Define a threshold for anomaly detection
threshold = np.percentile(probabilities, 5)  # for example, lowest 5% as anomalies

# Classify points as anomalies or normal based on the threshold
anomalies = probabilities < threshold

# Add the anomaly labels to the original DataFrame for easy reference
data_dropped['Anomaly'] = anomalies

# Plotting (pair plot for 2-dimensional view if many numerical columns)
# Here, we're plotting only the first two columns for visualization purposes
plt.figure(figsize=(10, 6))
plt.scatter(numerical_data[:, 0], numerical_data[:, 1], c=np.where(anomalies, 'red', 'blue'), label='Data')
plt.xlabel(data_dropped.select_dtypes(include=[np.number]).columns[0])
plt.ylabel(data_dropped.select_dtypes(include=[np.number]).columns[1])
plt.title('Anomaly Detection using Gaussian Mixture Model')
plt.legend(['Normal', 'Anomaly'])
plt.savefig('GMm.png')  # Save the figure as a PNG file

# Print the number of anomalies detected
num_anomalies = np.sum(anomalies)
print(f"Number of anomalies detected: {num_anomalies}")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors 
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# Prepare data
numeric_data = data_dropped.select_dtypes(include=['float64', 'int64']).values

# Isolation Forest
iso_forest = IsolationForest(n_estimators=100, contamination=0.08, random_state=42)
data_dropped['ISO_Anomaly'] = iso_forest.fit_predict(numeric_data) == -1  # True for anomalies

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
data_dropped['DBSCAN_Anomaly'] = dbscan.fit_predict(numeric_data) == -1  # True for anomalies

# KNN
nbrs = NearestNeighbors(n_neighbors=5).fit(numeric_data)
distances, _ = nbrs.kneighbors(numeric_data)
threshold = np.percentile(distances[:, 4], 95)
data_dropped['KNN_Anomaly'] = (distances[:, 4] > threshold)  # True for anomalies

# Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
gmm.fit(numeric_data)
probabilities = gmm.score_samples(numeric_data)
threshold_gmm = np.percentile(probabilities, 5)
data_dropped['GMM_Anomaly'] = probabilities < threshold_gmm  # True for anomalies

# K-Means for anomaly detection (e.g., with 2 clusters)
kmeans = KMeans(n_clusters=2, random_state=42)
data_dropped['KMeans_Cluster'] = kmeans.fit_predict(numeric_data)
# Assuming anomalies are in the smaller cluster
anomaly_cluster = data_dropped['KMeans_Cluster'].value_counts().idxmin()
data_dropped['KMeans_Anomaly'] = data_dropped['KMeans_Cluster'] == anomaly_cluster

# List of methods and their anomaly columns
methods = [
    ('Isolation Forest', 'ISO_Anomaly'),
    ('DBSCAN', 'DBSCAN_Anomaly'),
    ('KNN (LOF)', 'KNN_Anomaly'),
    ('Gaussian Mixture Model', 'GMM_Anomaly'),
    ('K-Means', 'KMeans_Anomaly')
]

# Plot individual method results
fig, axs = plt.subplots(2, 3, figsize=(18, 12))
for i, (method_name, anomaly_column) in enumerate(methods):
    ax = axs[i // 3, i % 3]
    normal_data = data_dropped[data_dropped[anomaly_column] == False]
    anomalies = data_dropped[data_dropped[anomaly_column] == True]
    ax.scatter(normal_data['Number of Services'], normal_data['Average Medicare Payment Amount'],
               c='blue', label='Normal', alpha=0.5)
    ax.scatter(anomalies['Number of Services'], anomalies['Average Medicare Payment Amount'],
               color='red', label='Anomaly', alpha=0.7)
    ax.set_title(f'{method_name} Anomaly Detection')
    ax.set_xlabel('Number of Services')
    ax.set_ylabel('Average Medicare Payment Amount')
    ax.legend()

# Overlap Plot for All Method Combinations
import itertools

# Generate all unique 2-method combinations for overlap plots
method_combinations = list(itertools.combinations(methods, 2))
fig, axs = plt.subplots(4, 3, figsize=(18, 20))
for i, ((method1, col1), (method2, col2)) in enumerate(method_combinations):
    row, col = divmod(i, 3)
    combined_anomalies = data_dropped[[col1, col2]].any(axis=1)
    normal_combined = data_dropped[~combined_anomalies]
    anomalous_combined = data_dropped[combined_anomalies]
    
    ax = axs[row, col]
    ax.scatter(normal_combined['Number of Services'], normal_combined['Average Medicare Payment Amount'],
               c='green', label='Normal', alpha=0.5)
    ax.scatter(anomalous_combined['Number of Services'], anomalous_combined['Average Medicare Payment Amount'],
               color='purple', label='Anomaly', alpha=0.7)
    ax.set_title(f'Overlap of {method1} and {method2} Anomalies')
    ax.set_xlabel('Number of Services')
    ax.set_ylabel('Average Medicare Payment Amount')
    ax.legend()

plt.tight_layout()
plt.savefig('summary.png')  # Save the figure as a PNG file


# Summary Report
print("Anomaly Detection Summary:")
for method_name, anomaly_column in methods:
    count = data_dropped[anomaly_column].sum()
    print(f"{method_name}: {count} anomalies detected")
combined_anomalies_any = data_dropped[[col for _, col in methods]].any(axis=1)
combined_count_any = combined_anomalies_any.sum()
print(f"Combined (detected by any method): {combined_count_any} anomalies")

# Anomalies detected by all methods simultaneously
combined_anomalies_all = data_dropped[[col for _, col in methods]].all(axis=1)
combined_count_all = combined_anomalies_all.sum()
print(f"Combined (detected by all methods): {combined_count_all} anomalies")
# Count overlaps between methods
print("\nOverlap Anomalies Summary:")

# Create a DataFrame to track overlaps
overlap_summary = pd.DataFrame()

for method1, col1 in methods:
    for method2, col2 in methods:
        if method1 != method2:
            overlap_count = data_dropped[col1 & col2].sum()
            overlap_summary.loc[method1, method2] = overlap_count

# Display the overlap summary
print(overlap_summary)

# Count anomalies detected by all methods simultaneously
combined_anomalies_all = data_dropped[[col for _, col in methods]].all(axis=1)
combined_count_all = combined_anomalies_all.sum()
print(f"\nAnomalies detected by all methods simultaneously: {combined_count_all}")

# Count overlaps between methods
print("\nOverlap Anomalies Summary:")

# Create a DataFrame to track overlaps
overlap_summary = pd.DataFrame(index=[method[0] for method in methods], 
                                 columns=[method[0] for method in methods])

for (method1, col1), (method2, col2) in itertools.combinations(methods, 2):
    overlap_count = data_dropped[col1] & data_dropped[col2]
    overlap_summary.loc[method1, method2] = overlap_count.sum()

# Display the overlap summary
print(overlap_summary)

# Count anomalies detected by all methods simultaneously
combined_anomalies_all = data_dropped[[col for _, col in methods]].all(axis=1)
combined_count_all = combined_anomalies_all.sum()
print(f"\nAnomalies detected by all methods simultaneously: {combined_count_all}")


























