
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# %matplotlib inline

df=pd.read_csv('Healthcare Providers.csv')
df.columns

num_cols = ["Number of Services","Number of Medicare Beneficiaries",
            "Number of Distinct Medicare Beneficiary/Per Day Services",
            "Average Medicare Allowed Amount","Average Submitted Charge Amount",
            "Average Medicare Payment Amount","Average Medicare Standardized Amount"]

def RemoveComma(x):
    return x.replace(",","")

for colm in num_cols:
    df[colm] = pd.to_numeric(df[colm].apply(lambda x: RemoveComma(x)))
df.dtypes

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cat_cols=[
 'Credentials of the Provider',
 'Gender of the Provider',
 'Entity Type of the Provider',
 'State Code of the Provider',
 'Country Code of the Provider',
 'Provider Type',
 'Medicare Participation Indicator',
 'Place of Service',
 'HCPCS Code',
 'HCPCS Description',
 'HCPCS Drug Indicator',]
df_label_encoded = df.copy()

for col in cat_cols:
    df_label_encoded[f'{col}_label'] = le.fit_transform(df[col].astype(str))

df_label_encoded.head()

data=df_label_encoded[[
       'Entity Type of the Provider_label', 'State Code of the Provider_label',
       'Country Code of the Provider_label',
       'Medicare Participation Indicator_label', 'Place of Service_label',
       'HCPCS Drug Indicator_label'
]+num_cols]
data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

pip install scikit-optimize

from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV
from sklearn.metrics import make_scorer, silhouette_score

x="Number of Services"
y= "Average Medicare Standardized Amount"

knn_lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
lof_anomalies = knn_lof.fit_predict(data_scaled)

df['LOF_Anomaly'] = lof_anomalies

plt.figure(figsize=(18, 10))

df[num_cols].boxplot(figsize=(18, 10))

lof_anomaly_indices = df[df['LOF_Anomaly'] == -1].index

for i, col in enumerate(num_cols, start=1):
    plt.scatter([i] * len(lof_anomaly_indices), df.loc[lof_anomaly_indices, col],
                color='red', alpha=0.5, label='LOF Anomaly' if i == 1 else "")

plt.title('Box Plot of Numerical Columns with LOF Anomalies Highlighted')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.grid()

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.show()

df['LOF_Anomaly'].value_counts()

isolation_forest = IsolationForest(contamination=0.05, random_state=42)
iso_labels = isolation_forest.fit_predict(data_scaled)
df['IF_Anomaly'] = iso_labels

plt.figure(figsize=(18, 10))

df[num_cols].boxplot(figsize=(18, 10))

lof_anomaly_indices = df[df['IF_Anomaly'] == -1].index

for i, col in enumerate(num_cols, start=1):
    plt.scatter([i] * len(lof_anomaly_indices), df.loc[lof_anomaly_indices, col],
                color='red', alpha=0.5, label='IF Anomaly' if i == 1 else "")

plt.title('Box Plot of Numerical Columns with IF Anomalies Highlighted')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.grid()

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.show()

df['IF_Anomaly'].value_counts()

dbscan = DBSCAN(eps=0.5, min_samples=10)
dbscan_labels = dbscan.fit_predict(data_scaled)

dbscan_labels = np.where(dbscan_labels == -1, -1, 1)

df['DB_Anomaly'] = dbscan_labels

plt.figure(figsize=(18, 10))

df[num_cols].boxplot(figsize=(18, 10))

lof_anomaly_indices = df[df['DB_Anomaly'] == -1].index

for i, col in enumerate(num_cols, start=1):
    plt.scatter([i] * len(lof_anomaly_indices), df.loc[lof_anomaly_indices, col],
                color='red', alpha=0.5, label='DBSCAN Anomaly' if i == 1 else "")

plt.title('Box Plot of Numerical Columns with DBSCAN Anomalies Highlighted')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.grid()

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.show()

df['DB_Anomaly'].value_counts()

one_class_svm = OneClassSVM(kernel="rbf", gamma=0.1, nu=0.05)
svm_labels = one_class_svm.fit_predict(data_scaled)

df['SVM_Anomaly'] = svm_labels
plt.figure(figsize=(18, 10))

df[num_cols].boxplot(figsize=(18, 10))

lof_anomaly_indices = df[df['SVM_Anomaly'] == -1].index

for i, col in enumerate(num_cols, start=1):
    plt.scatter([i] * len(lof_anomaly_indices), df.loc[lof_anomaly_indices, col],
                color='red', alpha=0.5, label='SVM Anomaly' if i == 1 else "")

plt.title('Box Plot of Numerical Columns with SVM Anomalies Highlighted')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.grid()

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.show()

df['SVM_Anomaly'].value_counts()

data['LOF_Anomaly'] = lof_anomalies
data['IF_Anomaly'] = iso_labels
data['DB_Anomaly'] = dbscan_labels
data['SVM_Anomaly'] = svm_labels

def plot_anomalies(anomalies, x, y, model_name):

    x_min, x_max = data[x].quantile(0.01), data[x].quantile(0.99)
    y_min, y_max = data[y].quantile(0.01), data[y].quantile(0.99)

    normal_points = data[anomalies == 1]
    anomaly_points = data[anomalies == -1]

    plt.figure(figsize=(10, 6))
    plt.scatter(normal_points[x], normal_points[y], color='blue', s=10, label='Normal')
    plt.scatter(anomaly_points[x], anomaly_points[y], color='red', s=10, label='Anomaly')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"Anomalies Detected by {model_name}")

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend()
    plt.show()

plot_anomalies(data['DB_Anomaly'], 'Number of Services', 'Average Medicare Standardized Amount', 'DBSCAN')

plot_anomalies(data['IF_Anomaly'], 'Number of Services', 'Average Medicare Standardized Amount', 'Isolation Forest')

plot_anomalies(data['LOF_Anomaly'], 'Number of Services', 'Average Medicare Standardized Amount', 'LOF')

plot_anomalies(data['SVM_Anomaly'], 'Number of Services', 'Average Medicare Standardized Amount', 'SVM')

from sklearn.metrics import pairwise_distances
import numpy as np

model_scores = {}

def calculate_intra_cluster_variance(model_name):
    normal_data = data[data[f'{model_name}_Anomaly'] == 1]
    centroid = normal_data.mean(axis=0)
    distances = pairwise_distances(normal_data, [centroid])
    return distances.mean()

for model_name in ['LOF', 'IF', 'DB', 'SVM']:
    intra_cluster_variance = calculate_intra_cluster_variance(model_name)
    model_scores[model_name] = {'Intra-cluster Variance': intra_cluster_variance}

def consistency_check(model_name):
    return (data[f'{model_name}_Anomaly'] == -1).mean() * 100  # % of anomalies

for model_name in ['LOF', 'IF', 'DB', 'SVM']:
    anomaly_percentage = consistency_check(model_name)
    model_scores[model_name]['Anomaly Percentage'] = anomaly_percentage

for model_name, scores in model_scores.items():
    print(f"{model_name}: Intra-cluster Variance = {scores['Intra-cluster Variance']:.2f}, Anomaly % = {scores['Anomaly Percentage']:.2f}%")

best_model = min(model_scores, key=lambda x: model_scores[x]['Intra-cluster Variance'] if not np.isnan(model_scores[x]['Intra-cluster Variance']) else float('inf'))
print(f"Best Model based on Intra-cluster Variance: {best_model}")

from sklearn.ensemble import IsolationForest
from sklearn.metrics import pairwise_distances
import pandas as pd

contamination = [0.04, 0.05, 0.06]
n_estimators = [50, 100, 150, 200]
max_samples = [0.1, 0.5, 1.0]

# # Function to calculate intra-cluster variance
# def calculate_intra_cluster_variance(model_name):
#     normal_data = data[data[f'{model_name}_Anomaly'] == 1]
#     centroid = normal_data.mean(axis=0)
#     distances = pairwise_distances(normal_data, [centroid])
#     return distances.mean()

# # Anomaly Consistency Check
# def consistency_check(model_name):
#     return (data[f'{model_name}_Anomaly'] == -1).mean() * 100  # % of anomalies

results = []
best_params = None
best_icv_score = float('inf')
best_anomaly_percentage = 0

for cont in contamination:
    for n_est in n_estimators:
        for max_samp in max_samples:
            model_name = f'IF_cont{cont}_n{n_est}_samp{max_samp}'

            isolation_forest = IsolationForest(n_estimators=n_est, max_samples=max_samp, contamination=cont, random_state=42)
            isolation_forest.fit(data_scaled)
            labels = isolation_forest.predict(data_scaled)
            data.loc[:, model_name + '_Anomaly'] = labels

            icv_score = calculate_intra_cluster_variance(model_name)
            anomaly_percentage = consistency_check(model_name)

            results.append({
                'contamination': cont,
                'n_estimators': n_est,
                'max_samples': max_samp,
                'icv_score': icv_score,
                'anomaly_percentage': anomaly_percentage
            })

            if (icv_score < best_icv_score) or
               (icv_score == best_icv_score and anomaly_percentage > best_anomaly_percentage):
                best_icv_score = icv_score
                best_anomaly_percentage = anomaly_percentage
                best_params = {'contamination': cont, 'n_estimators': n_est, 'max_samples': max_samp}

results_df = pd.DataFrame(results)

print("Best Parameters:", best_params)
print("Best ICV Score:", best_icv_score)
print("Best Anomaly Percentage:", best_anomaly_percentage)
print("All Results:\n", results_df)

