import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# Load your data (make sure this path is correct for your environment)
df = pd.read_csv("Cleaned_Healthcare Providers.csv")
numerical_columns = ['Number of Services', 'Number of Medicare Beneficiaries',
                     'Number of Distinct Medicare Beneficiary/Per Day Services',
                     'Average Medicare Allowed Amount', 'Average Submitted Charge Amount',
                     'Average Medicare Payment Amount', 'Average Medicare Standardized Amount']
data = df[numerical_columns].copy()

# Anomaly detection and removal using IQR method
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

# Normalize and Standardize the dataset
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)
standardizer = StandardScaler()
standardized_data = standardizer.fit_transform(normalized_data)

# Train-test split (70-30)
X_train, X_test, y_train, y_test = train_test_split(standardized_data, np.ones(len(standardized_data)), test_size=0.3, random_state=42)

# Apply PCA for visualization purposes
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 1. Isolation Forest
iso_forest = IsolationForest(n_estimators=100, contamination=0.08, random_state=42)
iso_forest.fit(X_train)
y_pred_iso_forest = iso_forest.predict(X_test)
y_pred_iso_forest = np.where(y_pred_iso_forest == -1, 1, 0)

# 2. DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
y_pred_dbscan = dbscan.fit_predict(X_test)
y_pred_dbscan = np.where(y_pred_dbscan == -1, 1, 0)

# 3. KNN (LOF)
nbrs = NearestNeighbors(n_neighbors=5).fit(X_train)
distances, _ = nbrs.kneighbors(X_test)
threshold = np.percentile(distances[:, 4], 95)
y_pred_knn = (distances[:, 4] > threshold).astype(int)

# 4. Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
gmm.fit(X_train)
probabilities = gmm.score_samples(X_test)
threshold_gmm = np.percentile(probabilities, 5)
y_pred_gmm = (probabilities < threshold_gmm).astype(int)

# 5. K-Means Clustering (Anomalies in the smaller cluster)
kmeans = KMeans(n_clusters=2, random_state=42)
y_pred_kmeans = kmeans.fit_predict(X_test)
anomaly_cluster = np.argmin(np.bincount(y_pred_kmeans))
y_pred_kmeans = np.where(y_pred_kmeans == anomaly_cluster, 1, 0)

# Define methods and their corresponding anomaly predictions
methods = [
    ('Isolation Forest', y_pred_iso_forest),
    ('DBSCAN', y_pred_dbscan),
    ('KNN (LOF)', y_pred_knn),
    ('Gaussian Mixture Model', y_pred_gmm),
    ('K-Means', y_pred_kmeans)
]

# Step 1: Visualize the number of anomalies detected by each method
print("Anomaly Detection Summary:")
anomaly_counts = {}
for method_name, y_pred in methods:
    count = y_pred.sum()
    anomaly_counts[method_name] = count
    print(f"{method_name}: {count} anomalies detected")

# Plotting the number of anomalies detected by each method
plt.figure(figsize=(10, 6))
plt.bar(anomaly_counts.keys(), anomaly_counts.values(), color='skyblue')
plt.xlabel('Anomaly Detection Method')
plt.ylabel('Number of Anomalies Detected')
plt.title('Number of Anomalies Detected by Each Method')
plt.xticks(rotation=45, ha='right')
plt.show()

# Step 2: Visualize anomalies in 'Number of Services' vs. 'Number of Medicare Beneficiaries'
# These are the columns you are interested in
for method_name, y_pred in methods:
    # Create scatter plot for Number of Services vs. Number of Medicare Beneficiaries
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test[:, 0], X_test[:, 1], c='gray', label='Normal', alpha=0.5)
    plt.scatter(X_test[y_pred == 1, 0], X_test[y_pred == 1, 1], c='red', label='Anomaly', alpha=0.8)
    plt.xlabel('Number of Services')
    plt.ylabel('Number of Medicare Beneficiaries')
    plt.title(f'Anomalies Detected by {method_name}')
    plt.legend()
    plt.show()

# Step 2: Count overlaps between methods
overlap_counts = {}
for pair in itertools.combinations(methods, 2):
    method1, method2 = pair
    y_pred1 = method1[1]
    y_pred2 = method2[1]
    overlap = np.sum((y_pred1 == 1) & (y_pred2 == 1))
    overlap_counts[f"{method1[0]} & {method2[0]}"] = overlap

print("\nOverlap Counts Between Methods:")
for pair, count in overlap_counts.items():
    print(f"{pair}: {count} overlaps")

# Step 3: Model Evaluation for Random Anomaly
# Generate ground truth for evaluation (Here we assume that anomalies are labeled as 1, non-anomalies as 0)
np.random.seed(42)
ground_truth = np.random.choice([0, 1], size=len(X_test), p=[0.95, 0.05])

# Model Evaluation function
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=1)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=1)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=1)
    return accuracy, precision, recall, f1

# Evaluate each model on the test set
print("\nModel Evaluation on Test Set:")
for method_name, y_pred in methods:
    accuracy, precision, recall, f1 = evaluate_model(ground_truth, y_pred)
    
    print(f"{method_name} (Test Set Evaluation):")
    print(f"  Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

# Visualization for each model's anomalies (2D projection of PCA)
plt.figure(figsize=(12, 10))

# Visualize anomalies for each model in 2D
for i, (method_name, y_pred) in enumerate(methods):
    plt.subplot(3, 2, i+1)
    plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_pred, cmap='coolwarm', label=method_name, s=20)
    plt.title(f'{method_name} Anomalies')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')

plt.tight_layout()
plt.show()
