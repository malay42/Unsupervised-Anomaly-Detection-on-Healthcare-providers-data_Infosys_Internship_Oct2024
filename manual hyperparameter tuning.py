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
file_path = "C:/Users/shiva/Desktop/project/Cleaned_Healthcare_Providers.csv"
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

# ---------- Anomaly Detection (Pre-Tuning) ----------
# Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_labels = iso_forest.fit_predict(scaled_sampled_df)
tsne_df['Isolation_Forest_Pre'] = pca_df['Isolation_Forest_Pre'] = np.where(iso_labels == -1, 'Anomaly', 'Normal')

# One-Class SVM
svm_model = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.03)
svm_labels = svm_model.fit_predict(scaled_sampled_df)
tsne_df['OneClassSVM_Pre'] = pca_df['OneClassSVM_Pre'] = np.where(svm_labels == -1, 'Anomaly', 'Normal')

# Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
lof_labels = lof.fit_predict(scaled_sampled_df)
tsne_df['LocalOutlierFactor_Pre'] = pca_df['LocalOutlierFactor_Pre'] = np.where(lof_labels == -1, 'Anomaly', 'Normal')

# DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(scaled_sampled_df)
tsne_df['DBSCAN_Pre'] = pca_df['DBSCAN_Pre'] = np.where(dbscan_labels == -1, 'Anomaly', 'Normal')

# ---------- Hyperparameter Tuning ----------
# Fine-tuning Isolation Forest
iso_forest = IsolationForest(n_estimators=200, contamination=0.05, max_samples=0.8, random_state=42)
iso_labels = iso_forest.fit_predict(scaled_sampled_df)
tsne_df['Isolation_Forest_Post'] = pca_df['Isolation_Forest_Post'] = np.where(iso_labels == -1, 'Anomaly', 'Normal')

# Fine-tuning One-Class SVM
svm_model = OneClassSVM(kernel='rbf', gamma=0.005, nu=0.02)
svm_labels = svm_model.fit_predict(scaled_sampled_df)
tsne_df['OneClassSVM_Post'] = pca_df['OneClassSVM_Post'] = np.where(svm_labels == -1, 'Anomaly', 'Normal')

# Fine-tuning Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=15, contamination=0.05)
lof_labels = lof.fit_predict(scaled_sampled_df)
tsne_df['LocalOutlierFactor_Post'] = pca_df['LocalOutlierFactor_Post'] = np.where(lof_labels == -1, 'Anomaly', 'Normal')

# Fine-tuning DBSCAN
dbscan = DBSCAN(eps=1.2, min_samples=10)
dbscan_labels = dbscan.fit_predict(scaled_sampled_df)
tsne_df['DBSCAN_Post'] = pca_df['DBSCAN_Post'] = np.where(dbscan_labels == -1, 'Anomaly', 'Normal')

# ---------- Visualization ----------

# Function to plot a single t-SNE or PCA visualization
def plot_results(df, x_col, y_col, label_col, title):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x_col, y=y_col, hue=label_col, style=label_col, markers={'Normal': 'o', 'Anomaly': 'X'},
                    palette={'Normal': 'blue', 'Anomaly': 'red'}, data=df)
    plt.title(title)
    plt.legend(title=label_col, loc='upper right')
    plt.show()

# Pre-Tuning t-SNE
plot_results(tsne_df, 't-SNE1', 't-SNE2', 'Isolation_Forest_Pre', "Isolation Forest (t-SNE) - Pre-Tuning")
plot_results(tsne_df, 't-SNE1', 't-SNE2', 'OneClassSVM_Pre', "One-Class SVM (t-SNE) - Pre-Tuning")
plot_results(tsne_df, 't-SNE1', 't-SNE2', 'LocalOutlierFactor_Pre', "Local Outlier Factor (t-SNE) - Pre-Tuning")
plot_results(tsne_df, 't-SNE1', 't-SNE2', 'DBSCAN_Pre', "DBSCAN (t-SNE) - Pre-Tuning")

# Pre-Tuning PCA
plot_results(pca_df, 'PCA1', 'PCA2', 'Isolation_Forest_Pre', "Isolation Forest (PCA) - Pre-Tuning")
plot_results(pca_df, 'PCA1', 'PCA2', 'OneClassSVM_Pre', "One-Class SVM (PCA) - Pre-Tuning")
plot_results(pca_df, 'PCA1', 'PCA2', 'LocalOutlierFactor_Pre', "Local Outlier Factor (PCA) - Pre-Tuning")
plot_results(pca_df, 'PCA1', 'PCA2', 'DBSCAN_Pre', "DBSCAN (PCA) - Pre-Tuning")

# Post-Tuning t-SNE
plot_results(tsne_df, 't-SNE1', 't-SNE2', 'Isolation_Forest_Post', "Isolation Forest (t-SNE) - Post-Tuning")
plot_results(tsne_df, 't-SNE1', 't-SNE2', 'OneClassSVM_Post', "One-Class SVM (t-SNE) - Post-Tuning")
plot_results(tsne_df, 't-SNE1', 't-SNE2', 'LocalOutlierFactor_Post', "Local Outlier Factor (t-SNE) - Post-Tuning")
plot_results(tsne_df, 't-SNE1', 't-SNE2', 'DBSCAN_Post', "DBSCAN (t-SNE) - Post-Tuning")

# Post-Tuning PCA
plot_results(pca_df, 'PCA1', 'PCA2', 'Isolation_Forest_Post', "Isolation Forest (PCA) - Post-Tuning")
plot_results(pca_df, 'PCA1', 'PCA2', 'OneClassSVM_Post', "One-Class SVM (PCA) - Post-Tuning")
plot_results(pca_df, 'PCA1', 'PCA2', 'LocalOutlierFactor_Post', "Local Outlier Factor (PCA) - Post-Tuning")
plot_results(pca_df, 'PCA1', 'PCA2', 'DBSCAN_Post', "DBSCAN (PCA) - Post-Tuning")
