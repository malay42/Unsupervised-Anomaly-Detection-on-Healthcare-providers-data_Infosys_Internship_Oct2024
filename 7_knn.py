# KNN

from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the normalized dataset
file_path = "cleaned_data.csv"
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

# K-Nearest Neighbors (KNN) model
knn = NearestNeighbors(n_neighbors=5)
knn.fit(X)
distances, _ = knn.kneighbors(X)
# Compute the mean distance for each point to its neighbors
mean_distances = np.mean(distances, axis=1)
threshold = np.percentile(mean_distances, 95)  # Top 5% as anomalies

# Classify anomalies
df['Anomaly_KNN'] = (mean_distances > threshold).astype(int)

# Visualize anomalies detected by KNN
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=df['Anomaly_KNN'],
                palette={1: 'blue', 0: 'red'}, s=20, alpha=0.7)
plt.title("KNN Anomaly Detection")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend(title="Anomaly Detection (0: Normal, 1: Anomaly)")
plt.show()