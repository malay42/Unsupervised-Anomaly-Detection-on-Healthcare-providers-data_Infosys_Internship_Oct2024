from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the normalized dataset
file_path = "normalized_data.csv"  # Update to the correct path
df = pd.read_csv(file_path)

# Define the numeric columns for anomaly detection
numeric_columns = [
    'Number of Services', 'Number of Medicare Beneficiaries', 
    'Number of Distinct Medicare Beneficiary/Per Day Services',
    'Average Medicare Allowed Amount', 'Average Submitted Charge Amount',
    'Average Medicare Payment Amount', 'Average Medicare Standardized Amount'
]

# Prepare data for anomaly detection
X = df[numeric_columns].values

# Apply t-SNE for 2D visualization before anomaly detection
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plot original data distribution
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], color='blue', s=20, alpha=0.7)
plt.title("Data Distribution Before Anomaly Detection")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.show()










# Isolation Forest model
iso_forest = IsolationForest(contamination=0.05, random_state=42)
df['Anomaly_IsolationForest'] = iso_forest.fit_predict(X)

# Filter anomalies (-1 represents anomalies)
df_iso_cleaned = df[df['Anomaly_IsolationForest'] == 1]
X_iso_cleaned = df_iso_cleaned[numeric_columns].values

# Visualize anomalies detected by Isolation Forest
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=df['Anomaly_IsolationForest'],
                palette={1: 'blue', -1: 'red'}, s=20, alpha=0.7)
plt.title("Isolation Forest Anomaly Detection")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend(title="Anomaly Detection (1: Normal, -1: Anomaly)")
plt.show()




# One-Class SVM model
one_class_svm = OneClassSVM(gamma='scale', nu=0.05)
df['Anomaly_OneClassSVM'] = one_class_svm.fit_predict(X)

# Filter anomalies (-1 represents anomalies)
df_svm_cleaned = df[df['Anomaly_OneClassSVM'] == 1]
X_svm_cleaned = df_svm_cleaned[numeric_columns].values

# Visualize anomalies detected by One-Class SVM
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=df['Anomaly_OneClassSVM'],
                palette={1: 'blue', -1: 'red'}, s=20, alpha=0.7)
plt.title("One-Class SVM Anomaly Detection")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend(title="Anomaly Detection (1: Normal, -1: Anomaly)")
plt.show()





# KNN model for anomaly detection (distance-based)
knn = NearestNeighbors(n_neighbors=5)
knn.fit(X)
distances, indices = knn.kneighbors(X)

# Set a threshold distance for anomalies (e.g., 95th percentile)
threshold = np.percentile(distances[:, -1], 95)
df['Anomaly_KNN'] = np.where(distances[:, -1] > threshold, -1, 1)

# Filter anomalies
df_knn_cleaned = df[df['Anomaly_KNN'] == 1]
X_knn_cleaned = df_knn_cleaned[numeric_columns].values

# Visualize anomalies detected by KNN
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=df['Anomaly_KNN'],
                palette={1: 'blue', -1: 'red'}, s=20, alpha=0.7)
plt.title("KNN Anomaly Detection")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend(title="Anomaly Detection (1: Normal, -1: Anomaly)")
plt.show()







