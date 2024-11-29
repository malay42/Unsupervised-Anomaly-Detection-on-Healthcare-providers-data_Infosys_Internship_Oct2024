# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import pairwise_distances, calinski_harabasz_score
from sklearn.decomposition import PCA

# Function Definitions
def custom_score(estimator, X):
    """Compute Calinski-Harabasz score using PCA-reduced data."""
    labels = estimator.fit_predict(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    return calinski_harabasz_score(X_pca, labels)

def calculate_intra_cluster_variance(data, model_name):
    """Calculate average distance of normal points from the centroid."""
    normal_data = data[data[f'{model_name}_Anomaly'] == 1]
    centroid = normal_data.mean(axis=0)
    distances = pairwise_distances(normal_data, [centroid])
    return distances.mean()

def consistency_check(data, model_name):
    """Compute percentage of anomalies detected."""
    return (data[f'{model_name}_Anomaly'] == -1).mean() * 100

def calculate_calinski_harabasz(data, model_name):
    """Calculate Calinski-Harabasz score based on anomaly labels."""
    labels = data[f'{model_name}_Anomaly']
    if len(set(labels)) < 2:
        return np.nan
    features = data.drop(columns=[col for col in data.columns if '_Anomaly' in col])
    return calinski_harabasz_score(features, labels)

def plot_metric(models, values, title, ylabel, highlight_idx, default_color='blue'):
    """Plot model comparison metrics."""
    colors = ['red' if i == highlight_idx else default_color for i in range(len(models))]
    plt.figure(figsize=(8, 5))
    plt.bar(models, values, color=colors)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Models")
    plt.tight_layout()
    plt.show()


# Main Workflow
if __name__ == "__main__":
    data = pd.read_csv("labeled_data.csv")
    data.head()

    model_scores = {}
    model_names = ['LOF', 'IF', 'DB', 'SVM']

    for model_name in model_names:
        intra_cluster_variance = calculate_intra_cluster_variance(data, model_name)
        anomaly_percentage = consistency_check(data, model_name)
        calinski_harabasz = calculate_calinski_harabasz(data, model_name)
        model_scores[model_name] = {
            'Intra-cluster Variance': intra_cluster_variance,
            'Anomaly Percentage': anomaly_percentage,
            'Calinski-Harabasz Score': calinski_harabasz,
        }

    for model_name, scores in model_scores.items():
        print(
            f"{model_name}: "
            f"Intra-cluster Variance = {scores['Intra-cluster Variance']:.2f}, "
            f"Anomaly % = {scores['Anomaly Percentage']:.2f}%, "
            f"Calinski-Harabasz Score = {scores['Calinski-Harabasz Score']:.2f}"
        )

    best_model_variance = min(
        model_scores,
        key=lambda x: model_scores[x]['Intra-cluster Variance']
        if not np.isnan(model_scores[x]['Intra-cluster Variance'])
        else float('inf')
    )
    best_model_calinski = max(
        model_scores,
        key=lambda x: model_scores[x]['Calinski-Harabasz Score']
        if not np.isnan(model_scores[x]['Calinski-Harabasz Score'])
        else float('-inf')
    )

    print(f"Best Model based on Intra-cluster Variance: {best_model_variance}")
    print(f"Best Model based on Calinski-Harabasz Score: {best_model_calinski}")
    
    models = list(model_scores.keys())
    intra_variance = [model_scores[model]['Intra-cluster Variance'] for model in models]
    anomaly_percentage = [model_scores[model]['Anomaly Percentage'] for model in models]
    calinski_scores = [model_scores[model]['Calinski-Harabasz Score'] for model in models]

    min_intra_variance_idx = np.argmin(intra_variance)
    max_anomaly_idx = np.argmax(anomaly_percentage)
    max_calinski_idx = np.argmax(calinski_scores)

    # Plot each metric using the function
    plot_metric(
        models,
        intra_variance,
        "Intra-cluster Variance (Lower is Better)",
        "Intra-cluster Variance",
        min_intra_variance_idx
    )

    plot_metric(
        models,
        anomaly_percentage,
        "Anomaly Percentage",
        "Anomaly Percentage (%)",
        max_anomaly_idx
    )

    plot_metric(
        models,
        calinski_scores,
        "Calinski-Harabasz Score (Higher is Better)",
        "Calinski-Harabasz Score",
        max_calinski_idx
    )
 
