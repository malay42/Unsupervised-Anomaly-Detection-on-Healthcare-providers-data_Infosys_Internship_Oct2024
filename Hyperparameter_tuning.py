import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from sklearn.metrics import calinski_harabasz_score

df = pd.read_csv('Updated_HealthCare.csv')

cols = ['Number of Services', 'Number of Medicare Beneficiaries', 
                  'Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 
                  'Average Medicare Payment Amount', 'Number of Distinct Medicare Beneficiary/Per Day Services',
                  'Average Medicare Standardized Amount']

#Implementing hyperparameter tuning of Isolation Forest using Calinski Harabasz Score
contamination_values = [0.05, 0.1, 0.15]
n_estimators_values = [50, 100, 150]
max_samples_values = [0.1, 0.5, 1.0]

results = []

for contamination in contamination_values:
    for n_estimators in n_estimators_values:
        for max_samples in max_samples_values:
            iso_forest = IsolationForest(n_estimators=n_estimators,
                                         contamination=contamination,
                                         max_samples=max_samples,
                                         random_state=42)
            df['anomaly'] = iso_forest.fit_predict(df[cols])

            df['anomaly_binary'] = np.where(df['anomaly'] == -1, 1, 0)

            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(df[cols])

            ch_score = calinski_harabasz_score(pca_data, df['anomaly_binary'])
            
            results.append({
                'contamination': contamination,
                'n_estimators': n_estimators,
                'max_samples': max_samples,
                'ch_score': ch_score
            })
            
results_df = pd.DataFrame(results)

best_params = results_df.loc[results_df['ch_score'].idxmax()]

print("Best parameters:")
print(best_params)
'''
Best parameters:
contamination        0.150000
n_estimators       150.000000
max_samples          0.100000
ch_score         10106.842914
'''
#PCA Projection of Isolation Forest using the best parameters
best_iso_forest = IsolationForest(
    n_estimators=int(best_params['n_estimators']),
    contamination=best_params['contamination'],
    max_samples=best_params['max_samples'],
    random_state=42
)
df['anomaly'] = best_iso_forest.fit_predict(df[cols])

pca = PCA(n_components=2)
pca_data = pca.fit_transform(df[cols])

df_pca = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
df_pca['anomaly'] = df['anomaly']

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='anomaly',
                palette={1: 'blue', -1: 'red'}, alpha=0.6)

plt.title('PCA Projection of Isolation Forest Anomalies (Best Parameters)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Anomaly', loc='upper right', labels=['Normal', 'Anomaly'])
plt.show()

#Implementing hyperparameter tuning of Local Outlier Factor using Calinski Harabasz Score
n_neighbors_values = [5, 10, 15]
contamination_values = [0.05, 0.1, 0.15]

results = []

for n_neighbors in n_neighbors_values:
    for contamination in contamination_values:
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)

        df['anomaly'] = lof.fit_predict(df[cols])

        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(df[cols])

        ch_score = calinski_harabasz_score(pca_data, df['anomaly'])

        results.append({
            'n_neighbors': n_neighbors,
            'contamination': contamination,
            'ch_score': ch_score
        })

results_df = pd.DataFrame(results)
best_params = results_df.loc[results_df['ch_score'].idxmax()]

print("Best parameters for LOF:")
print(best_params)
'''
Best parameters for LOF:
n_neighbors       5.000000
contamination     0.150000
ch_score         60.870575
'''
#PCA Projection of LOF using the best parameters
best_lof = LocalOutlierFactor(
    n_neighbors=int(best_params['n_neighbors']),
    contamination=best_params['contamination']
)

df['anomaly'] = best_lof.fit_predict(df[cols])
pca = PCA(n_components=2)
pca_data = pca.fit_transform(df[cols])

df_pca = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
df_pca['anomaly'] = df['anomaly'] 

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='anomaly',
                palette={-1: 'red', 1: 'blue'}, alpha=0.6)

plt.title('PCA Projection of LOF Anomalies (Best Parameters)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Anomaly', loc='upper right', labels=['Anomaly', 'Normal'])
plt.show()
