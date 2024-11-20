# Import necessary libraries
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA

# ============================
# Sample a Subset of Data
# ============================
final=pd.read_csv('final.csv')
final_subset = final.sample(frac=0.3, random_state=42)  # Adjust the fraction as needed

# ============================
# Objective Functions for Models
# ============================

# Isolation Forest
def objective_if(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_samples': trial.suggest_float('max_samples', 0.5, 1.0),
        'contamination': trial.suggest_float('contamination', 0.01, 0.2),
        'random_state': 42
    }
    model = IsolationForest(**params)
    model.fit(final_subset)
    predictions = model.predict(final_subset)
    return (predictions == -1).sum() / len(predictions)  # Percentage of anomalies

# Local Outlier Factor
def objective_lof(trial):
    params = {
        'n_neighbors': trial.suggest_int('n_neighbors', 5, 50),
        'contamination': trial.suggest_float('contamination', 0.01, 0.2),
        'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
    }
    model = LocalOutlierFactor(**params, novelty=True)
    model.fit(final_subset)
    predictions = model.predict(final_subset)
    return (predictions == -1).sum() / len(predictions)

# One-Class SVM
def objective_svm(trial):
    params = {
        'kernel': trial.suggest_categorical('kernel', ['rbf', 'sigmoid']),
        'gamma': trial.suggest_float('gamma', 1e-4, 1e-1, log=True),
        'nu': trial.suggest_float('nu', 0.01, 0.2)
    }
    model = OneClassSVM(**params)
    model.fit(final_subset)
    predictions = model.predict(final_subset)
    return (predictions == -1).sum() / len(predictions)

# Elliptic Envelope
def objective_ellipse(trial):
    params = {
        'contamination': trial.suggest_float('contamination', 0.01, 0.2),
        'support_fraction': trial.suggest_float('support_fraction', 0.5, 1.0)
    }
    model = EllipticEnvelope(**params)
    model.fit(final_subset)
    predictions = model.predict(final_subset)
    return (predictions == -1).sum() / len(predictions)

# KNN
def objective_knn(trial):
    params = {
        'n_neighbors': trial.suggest_int('n_neighbors', 5, 50),
        'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'chebyshev', 'minkowski'])
    }
    knn = NearestNeighbors(**params)
    knn.fit(final_subset)
    distances, _ = knn.kneighbors(final_subset)
    mean_distances = distances.mean(axis=1)
    threshold = np.percentile(mean_distances, 90)
    predictions = [1 if dist > threshold else -1 for dist in mean_distances]
    return (np.array(predictions) == 1).sum() / len(predictions)

# PCA
def objective_pca(trial):
    params = {
        'n_components': trial.suggest_int('n_components', 2, min(10, final_subset.shape[1]))
    }
    pca = PCA(**params)
    pca_transformed = pca.inverse_transform(pca.fit_transform(final_subset))
    reconstruction_error = np.mean((final_subset.values - pca_transformed) ** 2, axis=1)
    threshold = np.percentile(reconstruction_error, 85)
    predictions = [1 if e > threshold else -1 for e in reconstruction_error]
    return (np.array(predictions) == 1).sum() / len(predictions)

# ============================
# Run Optuna Studies
# ============================
studies = {
    "Isolation Forest": optuna.create_study(direction='maximize'),
    "Local Outlier Factor": optuna.create_study(direction='maximize'),
    "One-Class SVM": optuna.create_study(direction='maximize'),
    "Elliptic Envelope": optuna.create_study(direction='maximize'),
    "KNN": optuna.create_study(direction='maximize'),
    "PCA": optuna.create_study(direction='maximize')
}

studies["Isolation Forest"].optimize(objective_if, n_trials=20, timeout=600)
studies["Local Outlier Factor"].optimize(objective_lof, n_trials=20, timeout=600)
studies["One-Class SVM"].optimize(objective_svm, n_trials=20, timeout=600)
studies["Elliptic Envelope"].optimize(objective_ellipse, n_trials=20, timeout=600)
studies["KNN"].optimize(objective_knn, n_trials=20, timeout=600)
studies["PCA"].optimize(objective_pca, n_trials=20, timeout=600)

# ============================
# Print Best Parameters
# ============================
for model, study in studies.items():
    print(f"Best parameters for {model}:", study.best_params)

# ============================
# Apply Tuned Parameters
# ============================
# Isolation Forest
tuned_params_if = studies["Isolation Forest"].best_params
isolation_forest_tuned = IsolationForest(**tuned_params_if)
final['anomaly_iforest'] = isolation_forest_tuned.fit_predict(final)

# Local Outlier Factor
tuned_params_lof = studies["Local Outlier Factor"].best_params
lof_tuned = LocalOutlierFactor(**tuned_params_lof, novelty=True)
final['lof_anomaly'] = lof_tuned.fit_predict(final)

# Elliptic Envelope
tuned_params_ellipse = studies["Elliptic Envelope"].best_params
elliptic_env_tuned = EllipticEnvelope(**tuned_params_ellipse)
final['ellipse_anomaly'] = elliptic_env_tuned.fit_predict(final)

# KNN
tuned_params_knn = studies["KNN"].best_params
knn_tuned = NearestNeighbors(**tuned_params_knn)
knn_tuned.fit(final)
distances, _ = knn_tuned.kneighbors(final)
mean_distances = distances.mean(axis=1)
threshold = np.percentile(mean_distances, 90)
final['knn_anomaly'] = [1 if dist > threshold else -1 for dist in mean_distances]

# PCA
tuned_params_pca = studies["PCA"].best_params
pca_tuned = PCA(**tuned_params_pca)
pca_transformed = pca_tuned.inverse_transform(pca_tuned.fit_transform(final_subset))
reconstruction_error = np.mean((final_subset.values - pca_transformed) ** 2, axis=1)
threshold = np.percentile(reconstruction_error, 85)
final['pca_anomaly'] = -1
final.loc[final_subset.index, 'pca_anomaly'] = [1 if e > threshold else -1 for e in reconstruction_error]

# One-Class SVM
tuned_params_ocsvm = studies["One-Class SVM"].best_params
one_class_svm_tuned = OneClassSVM(**tuned_params_ocsvm)
final['svm_anomaly'] = one_class_svm_tuned.fit_predict(final)

# ============================
# Visualization
# ============================
# Count Anomalies Detected
num_anomalies = {
    "Isolation Forest": np.sum(final['anomaly_iforest'] == -1),
    "Local Outlier Factor": np.sum(final['lof_anomaly'] == -1),
    "Elliptic Envelope": np.sum(final['ellipse_anomaly'] == -1),
    "KNN": np.sum(final['knn_anomaly'] == 1),
    "PCA": np.sum(final['pca_anomaly'] == 1),
    "One-Class SVM": np.sum(final['svm_anomaly'] == -1)
}

# Bar Plot
plt.figure(figsize=(10, 6))
plt.bar(num_anomalies.keys(), num_anomalies.values(), color=['blue', 'orange', 'green', 'red', 'purple', 'cyan'])
plt.xlabel('Anomaly Detection Models')
plt.ylabel('Number of Anomalies Detected')
plt.title('Anomalies Detected by Different Models')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
