import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM

df = pd.read_csv('Updated_HealthCare.csv')

cols = ['Number of Services', 'Number of Medicare Beneficiaries', 
                  'Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 
                  'Average Medicare Payment Amount', 'Number of Distinct Medicare Beneficiary/Per Day Services',
                  'Average Medicare Standardized Amount']

#Implementing Isolation Forest
isolation_forest = IsolationForest(contamination=0.05, random_state=42)
df['Anomaly_IsolationForest'] = isolation_forest.fit_predict(df[cols])

df_isolation = df[df['Anomaly_IsolationForest'] != -1].copy()
df_isolation.head()

#PCA Projection of Isolation Forest
pca = PCA(n_components=2)
pca_data = pca.fit_transform(df[cols])

df_pca = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
df_pca['anomaly'] = df['Anomaly_IsolationForest']

plt.figure(figsize=(10,6))
sns.scatterplot(data = df_pca, x = 'PC1', y = 'PC2', hue='anomaly',
                palette={-1: 'red', 1: 'blue'}, alpha = 0.6)

plt.title('PCA Projection of Isolation Forest Anomalies')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Anomaly', loc='upper right', labels=['Normal', 'Anomaly'])
plt.show()

#Implementing Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors= 30)
df['anomaly_lof'] = lof.fit_predict(df[cols])

df_lof = df[df['anomaly_lof']!= -1].copy()
df_lof.head()

#PCA Projection of LOF
pca = PCA(n_components=2)
pca_data = pca.fit_transform(df[cols])

df_pca = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
df_pca['anomaly'] = df['anomaly_lof']

plt.figure(figsize=(10,6))
sns.scatterplot(data = df_pca, x = 'PC1', y = 'PC2', hue='anomaly',
                palette={-1: 'red', 1: 'blue'}, alpha = 0.6)

plt.title('PCA Projection of LOF Anomalies')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Anomaly', loc='upper right', labels=['Normal', 'Anomaly'])
plt.show()

#Implementing Gaussian Mixture Model
gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42) 
gmm.fit(df[cols])

scores = gmm.score_samples(df[cols])

threshold = np.percentile(scores, 5)
df['Anomaly_GMM'] = (scores < threshold).astype(int) 
df['Anomaly_GMM'] = df['Anomaly_GMM'].map({1: -1, 0: 1})
df_gmm = df[df['Anomaly_GMM'] == 1].copy()
df_gmm.head()

#PCA Projection of GMM
pca = PCA(n_components=2)
pca_data = pca.fit_transform(df[cols])

df_pca = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
df_pca['anomaly'] = df['Anomaly_GMM']

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='anomaly',
                palette={1: 'blue', -1: 'red'}, alpha=0.6)

plt.title('PCA Projection of GMM Anomalies')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Anomaly', loc='upper right', labels=['Normal', 'Anomaly'])
plt.show()

#Implementing One Class SVM
oc_svm = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')
oc_svm.fit(df[cols])

df['anomaly_ocsvm'] = oc_svm.predict(df[cols])

#PCA Projection of One Class SVM
pca = PCA(n_components=2)
pca_data = pca.fit_transform(df[cols])  

df_pca = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
df_pca['anomaly'] = df['anomaly_ocsvm']  

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='anomaly', palette={1: 'blue', -1: 'red'}, alpha=0.6)

plt.title('PCA Projection of One-Class SVM Anomalies')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Anomaly', loc='upper right', labels=['Normal', 'Anomaly'])
plt.show()

#Percentage of normal points detected by each method
gmm_normal_percentage = (df['Anomaly_GMM'] == 1).mean() * 100
isolation_forest_normal_percentage = (df['Anomaly_IsolationForest'] == 1).mean() * 100
lof_normal_percentage = (df['anomaly_lof'] == 1).mean() * 100
svm_normal_percentage = (df['anomaly_ocsvm'] == 1).mean() * 100

print(f"Percentage of normal values in GMM: {gmm_normal_percentage:.2f}%")
print(f"Percentage of normal values in Isolation Forest: {isolation_forest_normal_percentage:.2f}%")
print(f"Percentage of normal values in LOF: {lof_normal_percentage:.2f}%")
print(f"Percentage of normal values in One-Class SVM: {svm_normal_percentage:.2f}%")

#Plot of number of outliers detected by each method
outliers_counts = {
    'GMM': (df['Anomaly_GMM'] == -1).sum(),
    'Isolation Forest': (df['Anomaly_IsolationForest'] == -1).sum(),
    'LOF': (df['anomaly_lof'] == -1).sum(),
    'One-Class SVM': (df['anomaly_ocsvm'] == -1).sum()
}

plt.figure(figsize=(10, 6))
plt.bar(outliers_counts.keys(), outliers_counts.values(), color=['blue', 'green', 'red', 'purple'])
plt.title('Number of Outliers Detected by Each Anomaly Detection Method')
plt.xlabel('Anomaly Detection Method')
plt.ylabel('Number of Outliers')
plt.show()

#Number of overlapping normal points detected by each method
methods = ['anomaly_lof', 'anomaly_ocsvm', 'Anomaly_IsolationForest', 'Anomaly_GMM']
overlap_counts = []

for i in methods:
    for j in methods:
        if(j > i):
            count = len(df[(df[i] == 1) & (df[j] == 1)])
            print(f"Overlap between {i} and {j}: {count}/{len(df)} ")
            overlap_counts.append((f"{i} vs {j}", count))
            print()

#Plot of number of overlapping points detected by each method
overlap_df = pd.DataFrame(overlap_counts, columns=['Method pair', 'Overlap Count'])

plt.figure(figsize=(10, 6))
plt.bar(overlap_df['Method pair'], overlap_df['Overlap Count'], color='skyblue')

plt.xlabel('Method Pairs')
plt.ylabel('Number of Overlapping Data')
plt.title('Overlap of Normal Data Detected by Different Methods')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
