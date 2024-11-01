import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import category_encoders as ce

# Load the dataset
file_path = "C:/Users/shiva/Desktop/PROJECT 1/Cleaned_Healthcare_Providers.csv"
df = pd.read_csv(file_path)

# Select numeric columns for analysis
numeric_columns = [
    'Number of Services', 'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services',
    'Average Medicare Allowed Amount', 'Average Submitted Charge Amount',
    'Average Medicare Payment Amount', 'Average Medicare Standardized Amount'
]

# Standardize numeric columns
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Apply Binary Encoding for categorical features
binary_encoding_columns = ['Provider Type', 'Credentials of the Provider', 'Gender of the Provider', 'Entity Type of the Provider']
binary_encoder = ce.BinaryEncoder(cols=binary_encoding_columns)
df_encoded = binary_encoder.fit_transform(df)

# Further encode any remaining categorical columns
label_encoding_columns = ['Medicare Participation Indicator', 'Place of Service', 'HCPCS Code', 'HCPCS Drug Indicator']
for col in label_encoding_columns:
    df_encoded[col] = df_encoded[col].astype('category').cat.codes

# Drop any non-numeric columns and NaN values
df_encoded = df_encoded.apply(pd.to_numeric, errors='coerce').dropna(axis=1)

# Standardize the cleaned dataset
scaled_df = scaler.fit_transform(df_encoded)

# Sample the data for faster t-SNE and PCA processing
sample_size = 5000  # Adjust based on your dataset size
df_sampled = df_encoded.sample(n=sample_size, random_state=42)
scaled_sampled_df = scaler.fit_transform(df_sampled)

# Apply t-SNE with optimized parameters
tsne = TSNE(n_components=2, perplexity=20, max_iter=300, method='barnes_hut', random_state=42)
tsne_components = tsne.fit_transform(scaled_sampled_df)
tsne_df = pd.DataFrame(tsne_components, columns=['t-SNE1', 't-SNE2'])

# Apply PCA to the sampled data
pca = PCA(n_components=2, random_state=42)
pca_components = pca.fit_transform(scaled_sampled_df)
pca_df = pd.DataFrame(pca_components, columns=['PCA1', 'PCA2'])

# Perform anomaly detection algorithms on the sampled data
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_labels = iso_forest.fit_predict(scaled_sampled_df)
tsne_df['Isolation_Forest'] = pca_df['Isolation_Forest'] = np.where(iso_labels == -1, 'Anomaly', 'Normal')

svm_model = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.03)
svm_labels = svm_model.fit_predict(scaled_sampled_df)
tsne_df['OneClassSVM'] = pca_df['OneClassSVM'] = np.where(svm_labels == -1, 'Anomaly', 'Normal')

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
lof_labels = lof.fit_predict(scaled_sampled_df)
tsne_df['LocalOutlierFactor'] = pca_df['LocalOutlierFactor'] = np.where(lof_labels == -1, 'Anomaly', 'Normal')

dbscan = DBSCAN(eps=1.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(scaled_sampled_df)
tsne_df['DBSCAN'] = pca_df['DBSCAN'] = np.where(dbscan_labels == -1, 'Anomaly', 'Normal')

# Visualization of Anomaly Detection Results in t-SNE and PCA Space
fig, axes = plt.subplots(4, 2, figsize=(18, 24), sharex=False, sharey=False)
fig.suptitle("Optimized Anomaly Detection in t-SNE and PCA Spaces", fontsize=18)

# t-SNE Visualizations
sns.scatterplot(x='t-SNE1', y='t-SNE2', hue='Isolation_Forest', style='Isolation_Forest', markers={'Normal': 'o', 'Anomaly': 'X'},
                palette={'Normal': 'blue', 'Anomaly': 'red'}, data=tsne_df, ax=axes[0, 0])
axes[0, 0].set_title("Isolation Forest (t-SNE)")

sns.scatterplot(x='t-SNE1', y='t-SNE2', hue='OneClassSVM', style='OneClassSVM', markers={'Normal': 'o', 'Anomaly': 'X'},
                palette={'Normal': 'green', 'Anomaly': 'red'}, data=tsne_df, ax=axes[1, 0])
axes[1, 0].set_title("One-Class SVM (t-SNE)")

sns.scatterplot(x='t-SNE1', y='t-SNE2', hue='LocalOutlierFactor', style='LocalOutlierFactor', markers={'Normal': 'o', 'Anomaly': 'X'},
                palette={'Normal': 'purple', 'Anomaly': 'red'}, data=tsne_df, ax=axes[2, 0])
axes[2, 0].set_title("Local Outlier Factor (t-SNE)")

sns.scatterplot(x='t-SNE1', y='t-SNE2', hue='DBSCAN', style='DBSCAN', markers={'Normal': 'o', 'Anomaly': 'X'},
                palette={'Normal': 'orange', 'Anomaly': 'red'}, data=tsne_df, ax=axes[3, 0])
axes[3, 0].set_title("DBSCAN (t-SNE)")

# PCA Visualizations
sns.scatterplot(x='PCA1', y='PCA2', hue='Isolation_Forest', style='Isolation_Forest', markers={'Normal': 'o', 'Anomaly': 'X'},
                palette={'Normal': 'blue', 'Anomaly': 'red'}, data=pca_df, ax=axes[0, 1])
axes[0, 1].set_title("Isolation Forest (PCA)")

sns.scatterplot(x='PCA1', y='PCA2', hue='OneClassSVM', style='OneClassSVM', markers={'Normal': 'o', 'Anomaly': 'X'},
                palette={'Normal': 'green', 'Anomaly': 'red'}, data=pca_df, ax=axes[1, 1])
axes[1, 1].set_title("One-Class SVM (PCA)")

sns.scatterplot(x='PCA1', y='PCA2', hue='LocalOutlierFactor', style='LocalOutlierFactor', markers={'Normal': 'o', 'Anomaly': 'X'},
                palette={'Normal': 'purple', 'Anomaly': 'red'}, data=pca_df, ax=axes[2, 1])
axes[2, 1].set_title("Local Outlier Factor (PCA)")

sns.scatterplot(x='PCA1', y='PCA2', hue='DBSCAN', style='DBSCAN', markers={'Normal': 'o', 'Anomaly': 'X'},
                palette={'Normal': 'orange', 'Anomaly': 'red'}, data=pca_df, ax=axes[3, 1])
axes[3, 1].set_title("DBSCAN (PCA)")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Count anomalies for each detection method
iso_anomalies = np.sum(iso_labels == -1)
svm_anomalies = np.sum(svm_labels == -1)
lof_anomalies = np.sum(lof_labels == -1)
dbscan_anomalies = np.sum(dbscan_labels == -1)

# Print anomaly counts
print(f"Total Anomalies Detected by Isolation Forest: {iso_anomalies}")
print(f"Total Anomalies Detected by One-Class SVM: {svm_anomalies}")
print(f"Total Anomalies Detected by Local Outlier Factor: {lof_anomalies}")
print(f"Total Anomalies Detected by DBSCAN: {np.sum(dbscan_labels == -1)}")
# Perform anomaly detection on the entire dataset
iso_labels_full = iso_forest.fit_predict(scaled_df)
svm_labels_full = svm_model.fit_predict(scaled_df)
lof_labels_full = lof.fit_predict(scaled_df)
dbscan_labels_full = dbscan.fit_predict(scaled_df)

# Count anomalies for each detection method on the full dataset
iso_anomalies_full = np.sum(iso_labels_full == -1)
svm_anomalies_full = np.sum(svm_labels_full == -1)
lof_anomalies_full = np.sum(lof_labels_full == -1)
dbscan_anomalies_full = np.sum(dbscan_labels_full == -1)

# Print anomaly counts for the full dataset
print(f"Total Anomalies Detected by Isolation Forest (Full Dataset): {iso_anomalies_full}")
print(f"Total Anomalies Detected by One-Class SVM (Full Dataset): {svm_anomalies_full}")
print(f"Total Anomalies Detected by Local Outlier Factor (Full Dataset): {lof_anomalies_full}")
print(f"Total Anomalies Detected by DBSCAN (Full Dataset): {dbscan_anomalies_full}")

import matplotlib.pyplot as plt

# Collect anomaly counts for both sampled data and full dataset
anomaly_counts = {
    "Method": ["Isolation Forest", "One-Class SVM", "Local Outlier Factor", "DBSCAN"],
    "Sampled Data": [iso_anomalies, svm_anomalies, lof_anomalies, dbscan_anomalies],
    "Full Dataset": [iso_anomalies_full, svm_anomalies_full, lof_anomalies_full, dbscan_anomalies_full]
}

# Convert to DataFrame for easier plotting
anomaly_df = pd.DataFrame(anomaly_counts)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
width = 0.35  # Width of the bars

# Plot bars for sampled data and full dataset
ax.bar(anomaly_df["Method"], anomaly_df["Sampled Data"], width, label="Sampled Data", color="skyblue")
ax.bar(anomaly_df["Method"], anomaly_df["Full Dataset"], width, bottom=anomaly_df["Sampled Data"], label="Full Dataset", color="salmon")

# Adding labels
ax.set_xlabel("Anomaly Detection Method")
ax.set_ylabel("Number of Anomalies Detected")
ax.set_title("Comparison of Anomalies Detected by Each Method (Sampled vs Full Dataset)")
ax.legend()

# Display the plot
plt.xticks(rotation=45)
plt.show()

import time
from sklearn.metrics import pairwise_distances

import time

# Data to store timing and scores for plotting
anomaly_counts_sampled = []
anomaly_counts_full = []
timings_sampled = []
timings_full = []
scores_sampled = []
scores_full = []

# Helper function for anomaly detection
def run_anomaly_detection(model, data, label, scores_list, counts_list, timings_list):
    start_time = time.time()
    labels = model.fit_predict(data)
    end_time = time.time()

    # Calculate outlier scores if available (e.g., for models that have decision_function or negative_outlier_factor_)
    if hasattr(model, 'decision_function'):
        scores = model.decision_function(data)
    elif hasattr(model, 'negative_outlier_factor_'):
        scores = -model.negative_outlier_factor_
    else:
        scores = np.nan * np.ones(len(data))  # Use NaN for models without scoring support, such as DBSCAN

    scores_list.append(scores)
    counts_list.append(np.sum(labels == -1))
    timings_list.append(end_time - start_time)
    print(f"{label} - Anomalies Detected: {np.sum(labels == -1)}, Time Taken: {end_time - start_time:.2f} seconds")

# Run anomaly detection on sampled data
print("Running on Sampled Data:")
run_anomaly_detection(iso_forest, scaled_sampled_df, "Isolation Forest", scores_sampled, anomaly_counts_sampled, timings_sampled)
run_anomaly_detection(svm_model, scaled_sampled_df, "One-Class SVM", scores_sampled, anomaly_counts_sampled, timings_sampled)
run_anomaly_detection(lof, scaled_sampled_df, "Local Outlier Factor", scores_sampled, anomaly_counts_sampled, timings_sampled)
run_anomaly_detection(dbscan, scaled_sampled_df, "DBSCAN", scores_sampled, anomaly_counts_sampled, timings_sampled)

# Run anomaly detection on full dataset
print("\nRunning on Full Dataset:")
run_anomaly_detection(iso_forest, scaled_df, "Isolation Forest", scores_full, anomaly_counts_full, timings_full)
run_anomaly_detection(svm_model, scaled_df, "One-Class SVM", scores_full, anomaly_counts_full, timings_full)
run_anomaly_detection(lof, scaled_df, "Local Outlier Factor", scores_full, anomaly_counts_full, timings_full)
run_anomaly_detection(dbscan, scaled_df, "DBSCAN", scores_full, anomaly_counts_full, timings_full)

# Plotting the number of anomalies and timing comparison
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plotting the number of anomalies detected
axes[0].bar(anomaly_df["Method"], anomaly_counts_sampled, width=0.4, label="Sampled Data", color="skyblue")
axes[0].bar(anomaly_df["Method"], anomaly_counts_full, width=0.4, bottom=anomaly_counts_sampled, label="Full Dataset", color="salmon")
axes[0].set_title("Number of Anomalies Detected by Each Method")
axes[0].set_ylabel("Number of Anomalies")
axes[0].legend()

# Plotting execution timing
axes[1].bar(anomaly_df["Method"], timings_sampled, width=0.4, label="Sampled Data", color="skyblue")
axes[1].bar(anomaly_df["Method"], timings_full, width=0.4, bottom=timings_sampled, label="Full Dataset", color="salmon")
axes[1].set_title("Execution Time for Each Method")
axes[1].set_ylabel("Execution Time (seconds)")
axes[1].legend()

plt.suptitle("Comparison of Anomaly Detection Methods (Anomalies and Timing)")
plt.show()
