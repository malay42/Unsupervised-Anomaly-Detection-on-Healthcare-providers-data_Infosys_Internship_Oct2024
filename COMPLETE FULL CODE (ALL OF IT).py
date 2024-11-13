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
# from sklearn.mixture import GaussianMixture
# import numpy as np
# import matplotlib.pyplot as plt

# # Select only numerical columns from data_dropped
# numerical_data = data_dropped.select_dtypes(include=[np.number]).values

# # Fit Gaussian Mixture Model
# gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
# gmm.fit(numerical_data)

# # Get probabilities for each point
# probabilities = gmm.score_samples(numerical_data)

# # Define a threshold for anomaly detection
# threshold = np.percentile(probabilities, 5)  # for example, lowest 5% as anomalies

# # Classify points as anomalies or normal based on the threshold
# anomalies = probabilities < threshold

# # Add the anomaly labels to the original DataFrame for easy reference
# data_dropped['Anomaly'] = anomalies

# # Plotting (pair plot for 2-dimensional view if many numerical columns)
# # Here, we're plotting only the first two columns for visualization purposes
# plt.figure(figsize=(10, 6))
# plt.scatter(numerical_data[:, 0], numerical_data[:, 1], c=np.where(anomalies, 'red', 'blue'), label='Data')
# plt.xlabel(data_dropped.select_dtypes(include=[np.number]).columns[0])
# plt.ylabel(data_dropped.select_dtypes(include=[np.number]).columns[1])
# plt.title('Anomaly Detection using Gaussian Mixture Model')
# plt.legend(['Normal', 'Anomaly'])
# plt.savefig('GMm.png')  # Save the figure as a PNG file

# # Print the number of anomalies detected
# num_anomalies = np.sum(anomalies)
# print(f"Number of anomalies detected: {num_anomalies}")
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.ensemble import IsolationForest
# from sklearn.cluster import DBSCAN, KMeans
# from sklearn.neighbors import NearestNeighbors
# from sklearn.decomposition import PCA
# from sklearn.mixture import GaussianMixture

# # Select numeric columns
# numeric_data = data_dropped.select_dtypes(include=['float64', 'int64']).values

# # Isolation Forest
# iso_forest = IsolationForest(n_estimators=100, contamination=0.08, random_state=42)
# data_dropped['ISO_Anomaly'] = iso_forest.fit_predict(numeric_data) == -1  # True for anomalies
# # Calculate and add reconstruction error for Isolation Forest
# data_dropped['ISO_Reconstruction_Error'] = -iso_forest.decision_function(numeric_data)

# # DBSCAN
# dbscan = DBSCAN(eps=0.5, min_samples=5)
# data_dropped['DBSCAN_Anomaly'] = dbscan.fit_predict(numeric_data) == -1  # True for anomalies
# # Calculate and add reconstruction error for DBSCAN
# nbrs = NearestNeighbors(n_neighbors=5).fit(numeric_data)
# distances, _ = nbrs.kneighbors(numeric_data)
# dbscan_reconstruction_error = np.min(distances, axis=1)
# data_dropped['DBSCAN_Reconstruction_Error'] = dbscan_reconstruction_error

# # KNN
# threshold = np.percentile(distances[:, 4], 95)
# data_dropped['KNN_Anomaly'] = (distances[:, 4] > threshold)
# # Calculate and add reconstruction error for KNN
# data_dropped['KNN_Reconstruction_Error'] = distances[:, 4]

# # Gaussian Mixture Model
# gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
# gmm.fit(numeric_data)
# probabilities = gmm.score_samples(numeric_data)
# threshold_gmm = np.percentile(probabilities, 5)
# data_dropped['GMM_Anomaly'] = probabilities < threshold_gmm
# # Calculate and add reconstruction error for GMM
# data_dropped['GMM_Reconstruction_Error'] = -probabilities

# # K-Means
# kmeans = KMeans(n_clusters=2, random_state=42)
# data_dropped['KMeans_Cluster'] = kmeans.fit_predict(numeric_data)
# anomaly_cluster = data_dropped['KMeans_Cluster'].value_counts().idxmin()
# data_dropped['KMeans_Anomaly'] = data_dropped['KMeans_Cluster'] == anomaly_cluster
# # Calculate and add reconstruction error for K-Means
# kmeans_distances = np.linalg.norm(numeric_data - kmeans.cluster_centers_[data_dropped['KMeans_Cluster']], axis=1)
# data_dropped['KMeans_Reconstruction_Error'] = kmeans_distances

# # Visualize each method's anomalies
# methods = [
#     ('Isolation Forest', 'ISO_Anomaly'),
#     ('DBSCAN', 'DBSCAN_Anomaly'),
#     ('KNN (LOF)', 'KNN_Anomaly'),
#     ('Gaussian Mixture Model', 'GMM_Anomaly'),
#     ('K-Means', 'KMeans_Anomaly')
# ]

# fig, axs = plt.subplots(2, 3, figsize=(18, 12))
# for i, (method_name, anomaly_column) in enumerate(methods):
#     ax = axs[i // 3, i % 3]
#     normal_data = data_dropped[data_dropped[anomaly_column] == False]
#     anomalies = data_dropped[data_dropped[anomaly_column] == True]
#     ax.scatter(normal_data['Number of Services'], normal_data['Average Medicare Payment Amount'],
#                c='blue', label='Normal', alpha=0.5)
#     ax.scatter(anomalies['Number of Services'], anomalies['Average Medicare Payment Amount'],
#                color='red', label='Anomaly', alpha=0.7)
#     ax.set_title(f'{method_name} Anomaly Detection')
#     ax.set_xlabel('Number of Services')
#     ax.set_ylabel('Average Medicare Payment Amount')
#     ax.legend()

# # Overlap of anomalies for each method pair
# import itertools
# method_combinations = list(itertools.combinations(methods, 2))
# fig, axs = plt.subplots(4, 3, figsize=(18, 20))
# for i, ((method1, col1), (method2, col2)) in enumerate(method_combinations):
#     row, col = divmod(i, 3)
#     combined_anomalies = data_dropped[[col1, col2]].any(axis=1)
#     normal_combined = data_dropped[~combined_anomalies]
#     anomalous_combined = data_dropped[combined_anomalies]
#     ax = axs[row, col]
#     ax.scatter(normal_combined['Number of Services'], normal_combined['Average Medicare Payment Amount'],
#                c='green', label='Normal', alpha=0.5)
#     ax.scatter(anomalous_combined['Number of Services'], anomalous_combined['Average Medicare Payment Amount'],
#                color='purple', label='Anomaly', alpha=0.7)
#     ax.set_title(f'Overlap of {method1} and {method2} Anomalies')
#     ax.set_xlabel('Number of Services')
#     ax.set_ylabel('Average Medicare Payment Amount')
#     ax.legend()

# plt.tight_layout()
# plt.show()

# # Summary of anomaly detection
# print("Anomaly Detection Summary:")
# for method_name, anomaly_column in methods:
#     count = data_dropped[anomaly_column].sum()
#     print(f"{method_name}: {count} anomalies detected")

# combined_anomalies_any = data_dropped[[col for _, col in methods]].any(axis=1)
# combined_count_any = combined_anomalies_any.sum()
# print(f"Combined (detected by any method): {combined_count_any} anomalies")

# combined_anomalies_all = data_dropped[[col for _, col in methods]].all(axis=1)
# combined_count_all = combined_anomalies_all.sum()
# print(f"Combined (detected by all methods): {combined_count_all} anomalies")


# # Reconstruction Error Summary
# print("\nReconstruction Error Summary:")
# for method_name, error_column in [
#     ('Isolation Forest', 'ISO_Reconstruction_Error'),
#     ('DBSCAN', 'DBSCAN_Reconstruction_Error'),
#     ('KNN (LOF)', 'KNN_Reconstruction_Error'),
#     ('Gaussian Mixture Model', 'GMM_Reconstruction_Error'),
#     ('K-Means', 'KMeans_Reconstruction_Error')
# ]:
#     print(f"{method_name}: Mean Reconstruction Error = {data_dropped[error_column].mean()}")
   
   
   
   #after installing scikit optimize


#BAYESIAN SEARCH CODE

# import pandas as pd
# import numpy as np
# from sklearn.ensemble import IsolationForest
# from skopt import BayesSearchCV


# numeric_data = data_dropped.select_dtypes(include=['float64', 'int64'])

# def custom_scorer(estimator, X):

#     y_pred = estimator.predict(X)
   
#     return np.mean(y_pred == 1)

# # Define the parameter space
# param_space = {
#     'n_estimators': (50, 200),
#     'contamination': (0.01, 0.1),  
# }


# iso_forest = IsolationForest(random_state=42)


# bayes_search = BayesSearchCV(
#     estimator=iso_forest,
#     search_spaces=param_space,
#     n_iter=30,
#     scoring=custom_scorer,
#     n_jobs=-1,
#     cv=3
# )


# bayes_search.fit(numeric_data)

# best_iso_forest = bayes_search.best_estimator_

# print("Best parameters found: ", bayes_search.best_params_)
# #testing isolation forest with the results produced using bayesian search

# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.ensemble import IsolationForest

# numeric_data = data_dropped.select_dtypes(include=['float64', 'int64'])


# iso_forest = IsolationForest(n_estimators=50, contamination=0.01, random_state=42)


# iso_forest.fit(numeric_data)


# data_dropped['Anomaly'] = iso_forest.predict(numeric_data)


# anomalies = data_dropped[data_dropped['Anomaly'] == -1]


# plt.figure(figsize=(12, 6))


# plt.scatter(data_dropped[data_dropped['Anomaly'] == 1]['Number of Services'],
#             data_dropped[data_dropped['Anomaly'] == 1]['Average Medicare Payment Amount'],
#             label='Normal', alpha=0.5)

# plt.scatter(anomalies['Number of Services'],
#             anomalies['Average Medicare Payment Amount'],
#             color='red', label='Anomaly', alpha=0.7)
# plt.title('Anomaly Detection: Number of Services vs Average Medicare Payment Amount')
# plt.xlabel('Number of Services')
# plt.ylabel('Average Medicare Payment Amount')
# plt.legend()
# plt.grid()
# plt.show()

# plt.figure(figsize=(12, 6))

# plt.scatter(data_dropped[data_dropped['Anomaly'] == 1]['Number of Medicare Beneficiaries'],
#             data_dropped[data_dropped['Anomaly'] == 1]['Average Medicare Allowed Amount'],
#             label='Normal', alpha=0.5)
# plt.scatter(anomalies['Number of Medicare Beneficiaries'],
#             anomalies['Average Medicare Allowed Amount'],
#             color='red', label='Anomaly', alpha=0.7)
# plt.title('Anomaly Detection: Number of Medicare Beneficiaries vs Average Medicare Allowed Amount')
# plt.xlabel('Number of Medicare Beneficiaries')
# plt.ylabel('Average Medicare Allowed Amount')
# plt.legend()
# plt.grid()
# plt.show()
# print("Anomalies detected:")
# print(anomalies)
# print(f"Total number of anomalies: {anomalies.shape[0]}")
# plt.savefig('Bayesian Search.png')  


# #RANDOMISED SEARCH CODE

# import pandas as pd
# from sklearn.ensemble import IsolationForest
# from sklearn.model_selection import RandomizedSearchCV
# import numpy as np


# numeric_data = data_dropped.select_dtypes(include=['float64', 'int64'])


# iso_forest = IsolationForest(random_state=42)

# param_dist = {
#     'n_estimators': [10, 50, 100, 150, 200],  
#     'max_samples': [0.5, 0.75, 1.0],         
#     'contamination': [0.01, 0.05, 0.1, 0.2],  
#     'max_features': [0.5, 0.75, 1.0]        
# }


# random_search = RandomizedSearchCV(
#     estimator=iso_forest,
#     param_distributions=param_dist,
#     n_iter=20,                  
#     scoring='accuracy',      
#     cv=5,                     
#     random_state=42,
#     n_jobs=-1                  
# )


# random_search.fit(numeric_data)

# best_iso_forest = random_search.best_estimator_
# best_params = random_search.best_params_

# print("Best parameters found:", best_params)
# print("Best estimator:", best_iso_forest)

# data_dropped['Anomaly'] = best_iso_forest.fit_predict(numeric_data)
# anomalies = data_dropped[data_dropped['Anomaly'] == -1]

# print(f"Total number of anomalies detected with best parameters: {anomalies.shape[0]}")

# #implementation with the produced parameters

# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.ensemble import IsolationForest

# numeric_data = data_dropped.select_dtypes(include=['float64', 'int64'])


# iso_forest = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)


# iso_forest.fit(numeric_data)


# data_dropped['Anomaly'] = iso_forest.predict(numeric_data)


# anomalies = data_dropped[data_dropped['Anomaly'] == -1]


# plt.figure(figsize=(12, 6))


# plt.scatter(data_dropped[data_dropped['Anomaly'] == 1]['Number of Services'],
#             data_dropped[data_dropped['Anomaly'] == 1]['Average Medicare Payment Amount'],
#             label='Normal', alpha=0.5)

# plt.scatter(anomalies['Number of Services'],
#             anomalies['Average Medicare Payment Amount'],
#             color='red', label='Anomaly', alpha=0.7)
# plt.title('Anomaly Detection: Number of Services vs Average Medicare Payment Amount')
# plt.xlabel('Number of Services')
# plt.ylabel('Average Medicare Payment Amount')
# plt.legend()
# plt.grid()
# plt.show()

# plt.figure(figsize=(12, 6))

# plt.scatter(data_dropped[data_dropped['Anomaly'] == 1]['Number of Medicare Beneficiaries'],
#             data_dropped[data_dropped['Anomaly'] == 1]['Average Medicare Allowed Amount'],
#             label='Normal', alpha=0.5)
# plt.scatter(anomalies['Number of Medicare Beneficiaries'],
#             anomalies['Average Medicare Allowed Amount'],
#             color='red', label='Anomaly', alpha=0.7)
# plt.title('Anomaly Detection: Number of Medicare Beneficiaries vs Average Medicare Allowed Amount')
# plt.xlabel('Number of Medicare Beneficiaries')
# plt.ylabel('Average Medicare Allowed Amount')
# plt.legend()
# plt.grid()

# plt.show()
# print("Anomalies detected:")
# print(anomalies)
# print(f"Total number of anomalies: {anomalies.shape[0]}")
# plt.savefig('Randomised Search.png')  

# #OPTUNA OPTIMISATION CODE 

# import optuna
# from sklearn.ensemble import IsolationForest
# import numpy as np
# from sklearn.metrics import f1_score, make_scorer
# from sklearn.model_selection import train_test_split

# X_train, X_test = train_test_split(numeric_data, test_size=0.2, random_state=42)


# def objective(trial):
#     n_estimators = trial.suggest_int("n_estimators", 50, 300)
#     max_samples = trial.suggest_float("max_samples", 0.1, 1.0)
#     max_features = trial.suggest_float("max_features", 0.5, 1.0)
#     contamination = trial.suggest_float("contamination", 0.001, 0.1)


#     model = IsolationForest(
#         n_estimators=n_estimators,
#         max_samples=max_samples,
#         max_features=max_features,
#         contamination=contamination,
#         random_state=42
#     )

#     model.fit(X_train)
#     y_pred = model.predict(X_train)
#     y_pred = np.where(y_pred == -1, 1, 0) 


#     outlier_count = np.sum(y_pred == 1)  
    
#     return -outlier_count  


# study = optuna.create_study(direction="minimize")  
# study.optimize(objective, n_trials=50)

# best_params = study.best_params
# print("Best parameters found:", best_params)


# #Checking with isolation forest


# best_model = IsolationForest(
#     n_estimators=best_params['n_estimators'],
#     max_samples=best_params['max_samples'],
#     max_features=best_params['max_features'],
#     contamination=best_params['contamination'],
#     random_state=42
# )


# best_model.fit(numeric_data)


# outliers = best_model.predict(numeric_data)
# outlier_count = np.sum(outliers == -1)  
# print("Number of outliers detected:", outlier_count)
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.ensemble import IsolationForest

# numeric_data = data_dropped.select_dtypes(include=['float64', 'int64'])


# iso_forest = IsolationForest(n_estimators=259, contamination=0.09, random_state=42)


# iso_forest.fit(numeric_data)


# data_dropped['Anomaly'] = iso_forest.predict(numeric_data)


# anomalies = data_dropped[data_dropped['Anomaly'] == -1]


# plt.figure(figsize=(12, 6))


# plt.scatter(data_dropped[data_dropped['Anomaly'] == 1]['Number of Services'],
#             data_dropped[data_dropped['Anomaly'] == 1]['Average Medicare Payment Amount'],
#             label='Normal', alpha=0.5)

# plt.scatter(anomalies['Number of Services'],
#             anomalies['Average Medicare Payment Amount'],
#             color='red', label='Anomaly', alpha=0.7)
# plt.title('Anomaly Detection: Number of Services vs Average Medicare Payment Amount')
# plt.xlabel('Number of Services')
# plt.ylabel('Average Medicare Payment Amount')
# plt.legend()
# plt.grid()
# plt.show()

# plt.figure(figsize=(12, 6))

# plt.scatter(data_dropped[data_dropped['Anomaly'] == 1]['Number of Medicare Beneficiaries'],
#             data_dropped[data_dropped['Anomaly'] == 1]['Average Medicare Allowed Amount'],
#             label='Normal', alpha=0.5)
# plt.scatter(anomalies['Number of Medicare Beneficiaries'],
#             anomalies['Average Medicare Allowed Amount'],
#             color='red', label='Anomaly', alpha=0.7)
# plt.title('Anomaly Detection: Number of Medicare Beneficiaries vs Average Medicare Allowed Amount')
# plt.xlabel('Number of Medicare Beneficiaries')
# plt.ylabel('Average Medicare Allowed Amount')
# plt.legend()
# plt.grid()
# plt.savefig('Optuna.png')  
# plt.show()
# print("Anomalies detected:")
# print(anomalies)
# print(f"Total number of anomalies: {anomalies.shape[0]}")


# #MANUAL SEARCH CODE

# import numpy as np
# import pandas as pd
# from sklearn.ensemble import IsolationForest



# n_estimators = [50, 100, 200, 300]
# max_samples = [0.1, 0.5, 0.75, 1.0]
# max_features = [0.5, 0.75, 1.0]
# contamination = [0.001, 0.01, 0.1]


# results = []


# for n in n_estimators:
#     for samples in max_samples:
#         for features in max_features:
#             for cont in contamination:
                
#                 model = IsolationForest(n_estimators=n,
#                                         max_samples=samples,
#                                         max_features=features,
#                                         contamination=cont,
#                                         random_state=42)

                
#                 model.fit(numeric_data)

                
#                 outlier_predictions = model.predict(numeric_data)

                
#                 num_outliers = np.sum(outlier_predictions == -1)

                
#                 results.append((n, samples, features, cont, num_outliers))

                
#                 print(f"Parameters: {n}, {samples}, {features}, {cont} => Number of outliers detected: {num_outliers}")


# results_df = pd.DataFrame(results, columns=['n_estimators', 'max_samples', 'max_features', 'contamination', 'num_outliers'])

# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.ensemble import IsolationForest

# numeric_data = data_dropped.select_dtypes(include=['float64', 'int64'])


# iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42 , max_samples = 0.5 ,  max_features=0.5)


# iso_forest.fit(numeric_data)


# data_dropped['Anomaly'] = iso_forest.predict(numeric_data)


# anomalies = data_dropped[data_dropped['Anomaly'] == -1]


# plt.figure(figsize=(12, 6))


# plt.scatter(data_dropped[data_dropped['Anomaly'] == 1]['Number of Services'],
#             data_dropped[data_dropped['Anomaly'] == 1]['Average Medicare Payment Amount'],
#             label='Normal', alpha=0.5)

# plt.scatter(anomalies['Number of Services'],
#             anomalies['Average Medicare Payment Amount'],
#             color='red', label='Anomaly', alpha=0.7)
# plt.title('Anomaly Detection: Number of Services vs Average Medicare Payment Amount')
# plt.xlabel('Number of Services')
# plt.ylabel('Average Medicare Payment Amount')
# plt.legend()
# plt.grid()
# plt.show()

# plt.figure(figsize=(12, 6))

# plt.scatter(data_dropped[data_dropped['Anomaly'] == 1]['Number of Medicare Beneficiaries'],
#             data_dropped[data_dropped['Anomaly'] == 1]['Average Medicare Allowed Amount'],
#             label='Normal', alpha=0.5)
# plt.scatter(anomalies['Number of Medicare Beneficiaries'],
#             anomalies['Average Medicare Allowed Amount'],
#             color='red', label='Anomaly', alpha=0.7)
# plt.title('Anomaly Detection: Number of Medicare Beneficiaries vs Average Medicare Allowed Amount')
# plt.xlabel('Number of Medicare Beneficiaries')
# plt.ylabel('Average Medicare Allowed Amount')
# plt.legend()
# plt.grid()
# plt.savefig('Manual Search.png')
# plt.show()
# print("Anomalies detected:")
# print(anomalies)
# print(f"Total number of anomalies: {anomalies.shape[0]}")


# #A General OVERVIEW OF ALL THE METHODS USED WTH ISOLATION FOREST


# import matplotlib.pyplot as plt


# methods = ['Bayesian Search', 'Random Search', 'Optuna Optimization', 'Manual Search']
# outliers_detected = [561, 561, 5046, 5603] 


# plt.figure(figsize=(10, 6))
# plt.bar(methods, outliers_detected, color=['blue', 'orange', 'green', 'red'])


# plt.title('Number of Outliers Detected by Different Methods', fontsize=14)
# plt.xlabel('Detection Methods', fontsize=12)
# plt.ylabel('Number of Outliers Detected', fontsize=12)
# plt.xticks(rotation=10)  


# plt.grid(axis='y')
# plt.tight_layout()
# plt.savefig('All Hypertuning methods outlierd detected.png')
# plt.show()


# #MEAN ANOMALY SCORE COMPARISION

# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.ensemble import IsolationForest


# mean_anomaly_scores = {}


# hyperparameter_sets = {
#     "Bayesian Search": {"n_estimators": 50, "contamination": 0.01, "max_samples": 0.5, "max_features": 0.75},
#     "Random Search": {"n_estimators": 200, "contamination": 0.01, "max_samples": 0.5, "max_features": 0.75},
#     "Optuna Optimization": {"n_estimators": 259, "contamination": 0.0999, "max_samples": 0.4699, "max_features": 0.8071},
#     "Manual Search": {"n_estimators": 100, "contamination": 0.01, "max_samples": 0.5, "max_features": 0.5}
# }

# for method, params in hyperparameter_sets.items():
    
#     iso_forest = IsolationForest(
#         n_estimators=params["n_estimators"],
#         contamination=params["contamination"],
#         max_samples=params["max_samples"],
#         max_features=params["max_features"],
#         random_state=42
#     )
#     iso_forest.fit(numeric_data)
    

#     data_dropped['Anomaly'] = iso_forest.predict(numeric_data)
#     data_dropped['Anomaly_Score'] = iso_forest.decision_function(numeric_data)
#     outliers_scores = data_dropped[data_dropped['Anomaly'] == -1]['Anomaly_Score']
    
    
#     mean_anomaly_scores[method] = outliers_scores.mean()


# plt.figure(figsize=(10, 6))
# plt.bar(mean_anomaly_scores.keys(), mean_anomaly_scores.values(), color=['blue', 'orange', 'green', 'red'])
# plt.title("Mean Anomaly Score Comparison across Hyperparameter Tuning Methods")
# plt.xlabel("Hyperparameter Tuning Methods")
# plt.ylabel("Mean Anomaly Score for Outliers")
# plt.xticks(rotation=45)
# plt.grid(axis='y')
# plt.savefig('Mean anomaly comparision.png')
# plt.show()


# print("Mean Anomaly Scores for Outliers by Method:")
# for method, score in mean_anomaly_scores.items():
#     print(f"{method}: {score:.4f}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns


df = data_dropped



numeric_columns = [
    'Number of Services', 'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services',
    'Average Medicare Allowed Amount', 'Average Submitted Charge Amount',
    'Average Medicare Payment Amount', 'Average Medicare Standardized Amount'
]


scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])


X = df[numeric_columns].values


input_dim = X.shape[1]
encoding_dim = 4

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="relu")(input_layer)
encoder = Dropout(0.2)(encoder)  
decoder = Dense(input_dim, activation="linear")(encoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')


history = autoencoder.fit(X, X, epochs=20, batch_size=32, shuffle=True, validation_split=0.2, verbose=1)


X_pred = autoencoder.predict(X)
reconstruction_error = np.mean(np.square(X - X_pred), axis=1)


iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_labels = iso_forest.fit_predict(reconstruction_error.reshape(-1, 1))


best_threshold = None
best_f1 = 0
thresholds = range(80, 100, 2)  


scores = []

for percentile in thresholds:
    anomaly_threshold = np.percentile(reconstruction_error, percentile)
    predicted_anomalies = (reconstruction_error > anomaly_threshold).astype(int)

    # Calculate accuracy metrics for the current threshold
    accuracy = accuracy_score(iso_labels == -1, predicted_anomalies)
    precision = precision_score(iso_labels == -1, predicted_anomalies)
    recall = recall_score(iso_labels == -1, predicted_anomalies)
    f1 = f1_score(iso_labels == -1, predicted_anomalies)

    scores.append({'Threshold Percentile': percentile, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1})

    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = anomaly_threshold


scores_df = pd.DataFrame(scores)
print(scores_df)


optimal_threshold = np.percentile(reconstruction_error, 94)


predicted_anomalies = (reconstruction_error > optimal_threshold).astype(int)


anomalies = df[predicted_anomalies == 1]
print("Detected Anomalies:")
print(anomalies)


accuracy = accuracy_score(iso_labels == -1, predicted_anomalies)
precision = precision_score(iso_labels == -1, predicted_anomalies)
recall = recall_score(iso_labels == -1, predicted_anomalies)
f1 = f1_score(iso_labels == -1, predicted_anomalies)

print(f"Optimal Threshold (94th Percentile): {optimal_threshold:.2f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


cm = confusion_matrix(iso_labels == -1, predicted_anomalies)
print(f"Confusion Matrix:\n{cm}")

# Visualization


plt.figure(figsize=(15, 6))

# Subplot 1: Training Loss Over Epochs
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss Over Epochs")

# Subplot 2: Reconstruction Error Distribution with Anomaly Threshold
plt.subplot(1, 2, 2)
plt.hist(reconstruction_error, bins=50, color="skyblue", alpha=0.7, label="Reconstruction Errors")
plt.axvline(optimal_threshold, color='red', linestyle='dashed', linewidth=2, label="Anomaly Threshold")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.legend()
plt.title("Reconstruction Error Distribution with Anomaly Threshold")


plt.tight_layout()
plt.savefig('Autoencoder distrubution.png')  
plt.show()

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix at 94th Percentile Threshold autoencoder")
plt.savefig('Confusion Matrix at 94th Percentile Threshold autoencoder.png')  
plt.show()


def check_anomaly(row_num, reconstruction_error, optimal_threshold):
    if reconstruction_error[row_num] > optimal_threshold:
        return "Anomaly"
    else:
        return "Normal"


row_num = int(input("Enter the row number to check for anomaly (0-based index): "))


if row_num < 0 or row_num >= len(df):
    print("Invalid row number. Please enter a number between 0 and", len(df)-1)
else:

    result = check_anomaly(row_num, reconstruction_error, optimal_threshold)
    print(f"Row {row_num} is: {result}")
  
  
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds_roc = roc_curve(iso_labels == -1, reconstruction_error)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.savefig('ROC CURVE AUTOECNODER.png')  
plt.show()


plt.figure(figsize=(12, 8))
for i, column in enumerate(numeric_columns[:4], 1): 
    plt.subplot(2, 2, i)
    sns.kdeplot(df[column][predicted_anomalies == 0], label="Normal", fill=True, color="blue", alpha=0.3)
    sns.kdeplot(df[column][predicted_anomalies == 1], label="Anomaly", fill=True, color="red", alpha=0.3)
    plt.title(f"Distribution of {column}")
    plt.legend()
plt.tight_layout()
plt.savefig('Autoencoder feature distrubution.png')  
plt.show()
  





