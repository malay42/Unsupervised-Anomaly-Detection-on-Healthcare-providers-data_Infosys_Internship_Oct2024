import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

numeric_data = data_dropped.select_dtypes(include=['float64', 'int64'])
#ISOLATION FOREST

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


#DB SCAN

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


#KNN method

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
plt.savefig('knn.png')

#GAUSSIAN MIXTURE


from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt


numerical_data = data_dropped.select_dtypes(include=[np.number]).values


gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
gmm.fit(numerical_data)


probabilities = gmm.score_samples(numerical_data)


threshold = np.percentile(probabilities, 5)  


anomalies = probabilities < threshold


data_dropped['Anomaly'] = anomalies

plt.figure(figsize=(10, 6))
plt.scatter(numerical_data[:, 0], numerical_data[:, 1], c=np.where(anomalies, 'red', 'blue'), label='Data')
plt.xlabel(data_dropped.select_dtypes(include=[np.number]).columns[0])
plt.ylabel(data_dropped.select_dtypes(include=[np.number]).columns[1])
plt.title('Anomaly Detection using Gaussian Mixture Model')
plt.legend(['Normal', 'Anomaly'])
plt.savefig('GMm.png')  


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
data_dropped['GMM_Anomaly'] = probabilities < threshold_gmm  


kmeans = KMeans(n_clusters=2, random_state=42)
data_dropped['KMeans_Cluster'] = kmeans.fit_predict(numeric_data)

anomaly_cluster = data_dropped['KMeans_Cluster'].value_counts().idxmin()
data_dropped['KMeans_Anomaly'] = data_dropped['KMeans_Cluster'] == anomaly_cluster


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


import itertools

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
plt.savefig('summary.png')  


# Summary Report
print("Anomaly Detection Summary:")
for method_name, anomaly_column in methods:
    count = data_dropped[anomaly_column].sum()
    print(f"{method_name}: {count} anomalies detected")
combined_anomalies_any = data_dropped[[col for _, col in methods]].any(axis=1)
combined_count_any = combined_anomalies_any.sum()
print(f"Combined (detected by any method): {combined_count_any} anomalies")


combined_anomalies_all = data_dropped[[col for _, col in methods]].all(axis=1)
combined_count_all = combined_anomalies_all.sum()
print(f"Combined (detected by all methods): {combined_count_all} anomalies")

print("\nOverlap Anomalies Summary:")


overlap_summary = pd.DataFrame()

for method1, col1 in methods:
    for method2, col2 in methods:
        if method1 != method2:
            overlap_count = data_dropped[col1 & col2].sum()
            overlap_summary.loc[method1, method2] = overlap_count


print(overlap_summary)

combined_anomalies_all = data_dropped[[col for _, col in methods]].all(axis=1)
combined_count_all = combined_anomalies_all.sum()
print(f"\nAnomalies detected by all methods simultaneously: {combined_count_all}")


print("\nOverlap Anomalies Summary:")


overlap_summary = pd.DataFrame(index=[method[0] for method in methods], 
                                 columns=[method[0] for method in methods])

for (method1, col1), (method2, col2) in itertools.combinations(methods, 2):
    overlap_count = data_dropped[col1] & data_dropped[col2]
    overlap_summary.loc[method1, method2] = overlap_count.sum()


print(overlap_summary)


combined_anomalies_all = data_dropped[[col for _, col in methods]].all(axis=1)
combined_count_all = combined_anomalies_all.sum()
print(f"\nAnomalies detected by all methods simultaneously: {combined_count_all}")
