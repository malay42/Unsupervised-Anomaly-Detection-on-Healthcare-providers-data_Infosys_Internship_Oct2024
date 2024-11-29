# Import necessary libraries
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.manifold import TSNE


def plot_anomalies(df, num_cols, anomaly_col, anomaly_label):
    """Plot numerical columns with anomalies highlighted."""
    plt.figure(figsize=(18, 10))
    df[num_cols].boxplot()
    anomaly_indices = df[df[anomaly_col] == -1].index

    for i, col in enumerate(num_cols, start=1):
        plt.scatter(
            [i] * len(anomaly_indices),
            df.loc[anomaly_indices, col],
            color='red',
            alpha=0.5,
            label=anomaly_label if i == 1 else ""
        )
    plt.title(f"Box Plot with {anomaly_label} Highlighted")
    plt.ylabel('Values')
    plt.xticks(rotation=45)
    plt.grid()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()


def tsne_scatter(features, anomalies, dimensions=2):
    """Create a t-SNE scatter plot to visualize anomalies."""
    features_embedded = TSNE(n_components=dimensions, random_state=42).fit_transform(features)
    plt.figure(figsize=(8, 8))
    plt.scatter(
        *zip(*features_embedded[np.where(anomalies == 1)]),
        marker='o', color='g', s=2, alpha=0.3, label='Normal'
    )
    plt.scatter(
        *zip(*features_embedded[np.where(anomalies == -1)]),
        marker='o', color='r', s=2, alpha=0.7, label='Anomaly'
    )
    plt.legend(loc='best')
    plt.title("t-SNE Visualization of Anomalies")
    plt.show()


def detect_anomalies(df, data_scaled, num_cols, model, col_name, label):
    """Apply anomaly detection model and visualize results."""
    model_labels = model.fit_predict(data_scaled)
    model_labels = np.where(model_labels == -1, -1, 1)
    df[col_name] = model_labels
    plot_anomalies(df, num_cols, col_name, label)
    tsne_scatter(data_scaled, model_labels)
    print(df[col_name].value_counts())


# Main Workflow
if __name__ == "__main__":
    
    df = pd.read_csv('data_encoded.csv')

    num_cols = [
        "Number of Services", "Number of Medicare Beneficiaries",
        "Number of Distinct Medicare Beneficiary/Per Day Services",
        "Average Medicare Allowed Amount", "Average Submitted Charge Amount",
        "Average Medicare Payment Amount", "Average Medicare Standardized Amount"
    ]

    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df[num_cols])

    # Local Outlier Factor (LOF)
    lof_model = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
    detect_anomalies(df, data_scaled, num_cols, lof_model, 'LOF_Anomaly', 'LOF Anomaly')

    # Isolation Forest
    iso_model = IsolationForest(contamination=0.05, random_state=42)
    detect_anomalies(df, data_scaled, num_cols, iso_model, 'IF_Anomaly', 'Isolation Forest Anomaly')

    # DBSCAN
    dbscan_model = DBSCAN(eps=0.5, min_samples=10)
    detect_anomalies(df, data_scaled, num_cols, dbscan_model, 'DBSCAN_Anomaly', 'DBSCAN Anomaly')

    # One-Class SVM
    svm_model = OneClassSVM(kernel="rbf", gamma=0.1, nu=0.05)
    detect_anomalies(df, data_scaled, num_cols, svm_model, 'SVM_Anomaly', 'SVM Anomaly')

     # Save the labeled DataFrame to a CSV file
    df.to_csv('labeled_data.csv', index=False)
    print("The labeled data has been saved to 'labeled_data.csv'.")
