# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from itertools import combinations

# =========================
# Load Dataset Function
# =========================
def load_data(filepath):
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        print("Error: The specified file was not found.")
        exit()

# =========================
# Isolation Forest
# =========================
def isolation_forest(data):
    # Isolation Forest
    IF = IsolationForest(n_estimators=10, contamination=0.20)
    data['anomaly_iforest'] = IF.fit_predict(data)
    return len(data[data['anomaly_iforest'] == -1])

# =========================
# Local Outlier Factor
# =========================
def local_outlier_factor(data):
    # Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=5, contamination=0.10)
    data['lof_anomaly'] = lof.fit_predict(data)
    return len(data[data['lof_anomaly'] == -1])

# =========================
# One-Class SVM
# =========================
def one_class_svm(data):
    # One-Class SVM
    oc_svm = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.05)
    data['svm_anomaly'] = oc_svm.fit_predict(data)
    return len(data[data['svm_anomaly'] == -1])

# =========================
# Elliptic Envelope
# =========================
def elliptic_envelope(data):
    # Elliptic Envelope
    elliptic_env = EllipticEnvelope(contamination=0.05, random_state=42)
    data['ellipse_anomaly'] = elliptic_env.fit_predict(data)
    return len(data[data['ellipse_anomaly'] == -1])

# =========================
# KNN
# =========================
def knn(data):
    # K-Nearest Neighbors (KNN)
    knn = NearestNeighbors(n_neighbors=10)
    knn.fit(data)
    distances, indices = knn.kneighbors(data)
    mean_distances = distances.mean(axis=1)
    threshold = np.percentile(mean_distances, 90)
    data['knn_anomaly'] = [1 if dist > threshold else -1 for dist in mean_distances]
    return len(data[data['knn_anomaly'] == 1])

# =========================
# PCA
# =========================
def pca(data):
    # Principal Component Analysis (PCA)
    pca = PCA(n_components=2)
    final_array = data.values
    pca.fit(final_array)
    pca_transformed = pca.inverse_transform(pca.transform(final_array))
    reconstruction_error = np.mean((final_array - pca_transformed) ** 2, axis=1)
    threshold = np.percentile(reconstruction_error, 85)
    data['pca_anomaly'] = [1 if e > threshold else -1 for e in reconstruction_error]
    avg_error = reconstruction_error.mean()
    return len(data[data['pca_anomaly'] == 1]), avg_error

# =========================
# Overlap and Metrics Calculation
# =========================
def calculate_overlap(anomaly_labels):
    anomaly_df = pd.DataFrame(anomaly_labels)
    overlap_results = {}
    for (model1, labels1), (model2, labels2) in combinations(anomaly_labels.items(), 2):
        overlap_count = np.sum((anomaly_df[model1] == -1) & (anomaly_df[model2] == -1))
        overlap_results[f"{model1} vs {model2}"] = overlap_count
    return overlap_results

def calculate_silhouette_scores(data, anomaly_labels):
    scores = {}
    X = data.values
    for model_name, labels in anomaly_labels.items():
        if len(set(labels)) > 1:
            scores[model_name] = silhouette_score(X, labels)
        else:
            scores[model_name] = np.nan
    return scores

def calculate_calinski_harabasz_scores(data, anomaly_labels):
    scores = {}
    X = data.values
    for model_name, labels in anomaly_labels.items():
        if len(set(labels)) > 1:
            scores[model_name] = calinski_harabasz_score(X, labels)
        else:
            scores[model_name] = np.nan
    return scores

def calculate_davies_bouldin_scores(data, anomaly_labels):
    scores = {}
    X = data.values
    for model_name, labels in anomaly_labels.items():
        if len(set(labels)) > 1:
            scores[model_name] = davies_bouldin_score(X, labels)
        else:
            scores[model_name] = np.nan
    return scores

# =========================
# Plotting Functions
# =========================
def plot_overlap(overlap_results):
    overlap_pairs = list(overlap_results.keys())
    overlap_counts = list(overlap_results.values())
    plt.figure(figsize=(10, 6))
    plt.bar(overlap_pairs, overlap_counts, color="lightcoral")
    plt.xlabel("Model Pair")
    plt.ylabel("Number of Overlapping Anomalies")
    plt.title("Overlapping Anomalies Between Detection Models")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_metric_scores(scores, title, ylabel, color):
    models = list(scores.keys())
    values = [score if score is not np.nan else 0 for score in scores.values()]
    plt.figure(figsize=(10, 6))
    plt.bar(models, values, color=color)
    plt.xlabel('Anomaly Detection Models')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()

# =========================
# Main Function
# =========================
def main(filepath):
    # Load data
    data = load_data(filepath)
    
    # Run anomaly detection models
    results = {}
    results["Isolation Forest"] = {"anomalies": isolation_forest(data)}
    results["Local Outlier Factor"] = {"anomalies": local_outlier_factor(data)}
    results["One-Class SVM"] = {"anomalies": one_class_svm(data)}
    results["Elliptic Envelope"] = {"anomalies": elliptic_envelope(data)}
    results["KNN"] = {"anomalies": knn(data)}
    pca_anomalies, avg_error = pca(data)
    results["PCA"] = {"anomalies": pca_anomalies, "average_reconstruction_error": avg_error}

    # Prepare anomaly labels
    anomaly_labels = {key: data[f"{key.split()[0].lower()}_anomaly"] for key in results.keys()}
    
    # Calculate overlap
    overlap_results = calculate_overlap(anomaly_labels)
    results.update({f"{pair} overlap": {"count": count} for pair, count in overlap_results.items()})
    
    # Print results
    print("Comparison Results:")
    for model, metrics in results.items():
        print(f"{model}: {metrics}")
    
    # Plot overlaps
    plot_overlap(overlap_results)

    # Calculate and plot metrics
    silhouette_scores = calculate_silhouette_scores(data, anomaly_labels)
    plot_metric_scores(silhouette_scores, "Silhouette Score Comparison", "Silhouette Score", "lightblue")
    
    calinski_scores = calculate_calinski_harabasz_scores(data, anomaly_labels)
    plot_metric_scores(calinski_scores, "Calinski-Harabasz Score Comparison", "Calinski-Harabasz Score", "lightcoral")
    
    davies_bouldin_scores = calculate_davies_bouldin_scores(data, anomaly_labels)
    plot_metric_scores(davies_bouldin_scores, "Davies-Bouldin Score Comparison", "Davies-Bouldin Score (Lower is Better)", "lightgreen")

# Run main
if __name__ == "__main__":
    main('final.csv')
