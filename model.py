import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer, Normalizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score
import time

class AnomalyModel:
    def __init__(self, data):
        self.data = data
        self.labels = None

    def fit(self):
        raise NotImplementedError("Subclasses should implement this method")

    def get_results(self):
        return {
            'Execution Time': self.execution_time,
            'Anomalies Detected': np.sum(self.labels != 0)
        }

# KMeans Model
class KMeansModel(AnomalyModel):
    def fit(self, n_clusters=6):
        start_time = time.time()
        normalizer = Normalizer(norm='l2')
        data_scaled = normalizer.fit_transform(self.data)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.labels = kmeans.fit_predict(data_scaled)
        self.silhouette_score = silhouette_score(data_scaled, self.labels)
        self.execution_time = time.time() - start_time

    def get_results(self):
        results = super().get_results()
        results['Silhouette Score'] = self.silhouette_score
        return results

# DBSCAN Model
class DBSCANModel(AnomalyModel):
    def fit(self, eps=0.5, min_samples=5):
        start_time = time.time()
        normalizer = Normalizer(norm='l2')
        data_scaled = normalizer.fit_transform(self.data)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.labels = dbscan.fit_predict(data_scaled)
        self.silhouette_score = silhouette_score(data_scaled, self.labels) if len(set(self.labels)) > 1 else -1
        self.execution_time = time.time() - start_time

    def get_results(self):
        results = super().get_results()
        results['Silhouette Score'] = self.silhouette_score
        return results

# One-Class SVM Model
class OneClassSVMModel(AnomalyModel):
    def fit(self, kernel='rbf', gamma='auto', nu=0.1):
        start_time = time.time()
        normalizer = Normalizer(norm='l2')
        data_scaled = normalizer.fit_transform(self.data)
        svm = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
        self.labels = svm.fit_predict(data_scaled)
        self.execution_time = time.time() - start_time

# Isolation Forest Model
class IsolationForestModel(AnomalyModel):
    def fit(self, n_estimators=100, contamination=0.1):
        start_time = time.time()
        normalizer = Normalizer(norm='l2')
        data_scaled = normalizer.fit_transform(self.data)
        iso_forest = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
        self.labels = iso_forest.fit_predict(data_scaled)
        self.execution_time = time.time() - start_time

# Gaussian Mixture Model
class GaussianMixtureModel(AnomalyModel):
    def fit(self, n_components=2):
        start_time = time.time()
        normalizer = Normalizer(norm='l2')
        data_scaled = normalizer.fit_transform(self.data)
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        self.labels = gmm.fit_predict(data_scaled)
        self.silhouette_score = silhouette_score(data_scaled, self.labels)
        self.execution_time = time.time() - start_time

    def get_results(self):
        results = super().get_results()
        results['Silhouette Score'] = self.silhouette_score
        return results

# Local Outlier Factor (LOF) Model
class LocalOutlierFactorModel(AnomalyModel):
    def fit(self, n_neighbors=20, contamination=0.1):
        start_time = time.time()
        normalizer = Normalizer(norm='l2')
        data_scaled = normalizer.fit_transform(self.data)
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        self.labels = lof.fit_predict(data_scaled)
        self.execution_time = time.time() - start_time
        
class ModelComparison:
    def __init__(self, data):
        self.data = data
        self.results = {}

    def compare_models(self):
        models = {
            'K-Means': KMeansModel(self.data),
            'DBSCAN': DBSCANModel(self.data),
            'One-Class SVM': OneClassSVMModel(self.data),
            'Isolation Forest': IsolationForestModel(self.data),
            'Gaussian Mixture': GaussianMixtureModel(self.data),
            'KNN (LOF)': LocalOutlierFactorModel(self.data)
        }

        for model_name, model in models.items():
            model.fit()
            self.results[model_name] = model.get_results()
            print(f"{model_name}: {self.results[model_name]}")

    def plot_comparisons(self):
        labels = list(self.results.keys())
        execution_times = [metrics['Execution Time'] for metrics in self.results.values()]
        anomalies_detected = [metrics['Anomalies Detected'] for metrics in self.results.values()]

        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.barh(labels, execution_times, color='lightblue')
        plt.title('Model Execution Time Comparison')
        plt.xlabel('Time (seconds)')

        plt.subplot(1, 2, 2)
        plt.barh(labels, anomalies_detected, color='lightgreen')
        plt.title('Anomalies Detected Comparison')
        plt.xlabel('Number of Anomalies')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    path = 'cleaned_healthcare.csv'
    data = pd.read_csv(path)
    numerical_columns = [
        'Number of Services',
        'Number of Medicare Beneficiaries',
        'Number of Distinct Medicare Beneficiary/Per Day Services',
        'Average Medicare Allowed Amount',
        'Average Submitted Charge Amount',
        'Average Medicare Payment Amount',
        'Average Medicare Standardized Amount'
    ]
    for col in numerical_columns:
        data[col] = pd.to_numeric(data[col].str.replace(',', ''), errors='coerce')
    data = data[numerical_columns].dropna()

    comparison = ModelComparison(data)
    comparison.compare_models()
    comparison.plot_comparisons()
