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
from sklearn.metrics import silhouette_score

df = pd.read_csv('Updated_HealthCare.csv')

cols = ['Number of Services', 'Number of Medicare Beneficiaries', 
                  'Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 
                  'Average Medicare Payment Amount', 'Number of Distinct Medicare Beneficiary/Per Day Services',
                  'Average Medicare Standardized Amount']

#Implementing hyperparameter tuning of Isolation Forest using Calinski Harabasz Score
contamination_values = [0.03, 0.05, 0.07, 0.1]
n_estimators_values = [50, 100, 150, 200]
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

#Implementing hyperparameter tuning of GMM using Calinski Harabasz Score
n_components_values = [2, 3, 4]
covariance_type_values = ['full', 'tied', 'diag', 'spherical']

results = []

for n_components in n_components_values:
    for covariance_type in covariance_type_values:
        gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=42)
        gmm.fit(df[cols])

        probs = gmm.score_samples(df[cols])
        threshold = np.percentile(probs, 5)  
        df['anomaly'] = np.where(probs < threshold, -1, 1)

        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(df[cols])

        ch_score = calinski_harabasz_score(pca_data, df['anomaly'])
        results.append({
            'n_components': n_components,
            'covariance_type': covariance_type,
            'ch_score': ch_score
        })


results_df = pd.DataFrame(results)
best_params = results_df.loc[results_df['ch_score'].idxmax()]

print("Best parameters for GMM:")
print(best_params)
'''
Best parameters for GMM:
n_components                 2
covariance_type           diag
ch_score           6251.899491
'''
#PCA Projection of GMM using the best parameters
best_gmm = GaussianMixture(n_components=int(best_params['n_components']), covariance_type= best_params['covariance_type'], random_state=42)
best_gmm.fit(df[cols])

probs = best_gmm.score_samples(df[cols])
threshold = np.percentile(probs, 5)  
df['anomaly'] = np.where(probs < threshold, -1, 1)

pca = PCA(n_components=2)
pca_data = pca.fit_transform(df[cols])

df_pca = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
df_pca['anomaly'] = df['anomaly']
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='anomaly',
                palette={1: 'blue', -1: 'red'}, alpha=0.6)

plt.title('PCA Projection of GMM Anomalies (Best Parameters)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Anomaly', loc='upper right', labels=['Normal', 'Anomaly'])
plt.show()

#Implementing hyperparameter tuning for One Class GMM using Calinski Harabasz Score
nu_values = [0.01, 0.05, 0.1]
gamma_values = ['scale', 'auto']
kernel_values = ['rbf', 'linear', 'poly', 'sigmoid'] 

results = []

for nu in nu_values:
    for gamma in gamma_values:
        for kernel in kernel_values:
            oc_svm = OneClassSVM(nu=nu, gamma=gamma, kernel=kernel)
            df['anomaly'] = oc_svm.fit_predict(df[cols])

            df['anomaly_binary'] = np.where(df['anomaly'] == -1, 1, 0)

            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(df[cols])

            ch_score = calinski_harabasz_score(pca_data, df['anomaly_binary'])
            
            results.append({
                'nu': nu,
                'gamma': gamma,
                'kernel': kernel,
                'ch_score': ch_score
            })

results_df = pd.DataFrame(results)
best_params = results_df.loc[results_df['ch_score'].idxmax()]
print("Best parameters for One-Class SVM:")
print(best_params)
'''
Best parameters for One-Class SVM:
nu                  0.1
gamma             scale
kernel           linear
ch_score    7885.682315
'''
#PCA Projection for One Class GMM using the best parameters
best_oc_svm = OneClassSVM(nu=best_params['nu'], gamma=best_params['gamma'], kernel=best_params['kernel'])
df['anomaly'] = best_oc_svm.fit_predict(df[cols])

pca_data = pca.fit_transform(df[cols])
df_pca = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
df_pca['anomaly'] = df['anomaly'] 

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='anomaly', 
                palette={-1: 'red', 1: 'blue'}, alpha=0.6)

plt.title('PCA Projection of One-Class SVM Anomalies (Best Parameters)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Anomaly', loc='upper right', labels=['Normal','Anomaly'])
plt.show()

#Calculating Silhouette Score for each of the best parameters
df['isolation_forest_anomaly'] = best_iso_forest.predict(df[cols])
silhouette_isolation = silhouette_score(df[cols], df['isolation_forest_anomaly'])
print(f"Silhouette Score for Isolation Forest: {silhouette_isolation}")

df['lof_anomaly'] = best_lof.fit_predict(df[cols])
silhouette_lof = silhouette_score(df[cols], df['lof_anomaly'])
print(f"Silhouette Score for LOF: {silhouette_lof}")

df['gmm_anomaly'] = best_gmm.predict(df[cols])
silhouette_gmm = silhouette_score(df[cols], df['gmm_anomaly'])
print(f"Silhouette Score for GMM: {silhouette_gmm}")

df['svm_anomaly'] = best_oc_svm.predict(df[cols])
silhouette_svm = silhouette_score(df[cols], df['svm_anomaly'])
print(f"Silhouette Score for One-Class SVM: {silhouette_svm}")
'''
Silhouette Score for Isolation Forest: 0.3604482079664519
Silhouette Score for LOF: 0.0008583041812396687
Silhouette Score for GMM: 0.31432287685569577
Silhouette Score for One-Class SVM: 0.02956027019968757
'''
#Implementing hyperparameter tuning for Isolation Forest using Silhouette Score
contamination_values = [0.03, 0.05, 0.07, 0.1]
n_estimators_values = [50, 100, 150, 200]
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

            try:
                sil_score = silhouette_score(pca_data, df['anomaly_binary'])
                results.append({
                    'contamination': contamination,
                    'n_estimators': n_estimators,
                    'max_samples': max_samples,
                    'sil_score': sil_score
                })
            except ValueError:
                continue

results_df = pd.DataFrame(results)

best_params = results_df.loc[results_df['sil_score'].idxmax()]

print("Best parameters:")
print(best_params)
'''
Best parameters:
contamination      0.030000
n_estimators     200.000000
max_samples        0.100000
sil_score          0.383852
'''
#PCA Projection of Isolation Forest using the best parameters
best_iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.03,
    max_samples=0.1,
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

#Calculatng Calinski score for Isolation Forest for the best parameters detected by Silhouette Score
iso_forest = IsolationForest(n_estimators=200,
                                         contamination=0.03,
                                         max_samples=0.1,
                                         random_state=42)
df['anomaly'] = iso_forest.fit_predict(df[cols])

df['anomaly_binary'] = np.where(df['anomaly'] == -1, 1, 0)

pca = PCA(n_components=2)
pca_data = pca.fit_transform(df[cols])

ch_score = calinski_harabasz_score(pca_data, df['anomaly_binary'])

print(f"ch_score :{ch_score}")
#ch_score :3375.5100750875336
