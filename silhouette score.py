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
from sklearn.metrics import silhouette_score
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
sample_size = 5000  # Further reduce sample size
df_sampled = df_encoded.sample(n=sample_size, random_state=42)
scaled_sampled_df = scaler.fit_transform(df_sampled)

# Apply PCA to reduce dimensions before t-SNE (to 2 components)
pca = PCA(n_components=2, random_state=42)
pca_sampled_df = pca.fit_transform(scaled_sampled_df)

# Apply t-SNE with lower perplexity and adjusted max_iter
tsne = TSNE(n_components=2, perplexity=5, max_iter=300, random_state=42)
tsne_components = tsne.fit_transform(pca_sampled_df)
tsne_df = pd.DataFrame(tsne_components, columns=['t-SNE1', 't-SNE2'])

# ---------- Anomaly Detection ----------
models = {
    "Isolation Forest": IsolationForest(contamination=0.05, n_estimators=50, random_state=42),
    "One-Class SVM": OneClassSVM(nu=0.1, gamma='scale'),
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors=5, contamination=0.05),
    "DBSCAN": DBSCAN(eps=0.5, min_samples=5)
}

# Initialize a dictionary to store silhouette scores
silhouette_scores = {}

# Fit models and evaluate
for model_name, model in models.items():
    if model_name == "DBSCAN":
        # DBSCAN does not require fitting in the same way
        labels = model.fit_predict(scaled_sampled_df)
    else:
        labels = model.fit_predict(scaled_sampled_df)
    
    # Add labels to the DataFrame
    tsne_df[model_name] = np.where(labels == -1, 'Anomaly', 'Normal')
    
    # Compute silhouette score only for labeled data
    if model_name != "DBSCAN":
        score = silhouette_score(scaled_sampled_df, labels)
        silhouette_scores[model_name] = score
    else:
        silhouette_scores[model_name] = "N/A"  # DBSCAN doesn't return labels like others

# Print silhouette scores
print("Silhouette Scores for Anomaly Detection Models:")
for model_name, score in silhouette_scores.items():
    print(f"{model_name}: {score}")

# ---------- Visualization ----------
# Function to plot a single t-SNE visualization
def plot_results(df, x_col, y_col, label_col, title):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x_col, y=y_col, hue=label_col, style=label_col, markers={'Normal': 'o', 'Anomaly': 'X'},
                    palette={'Normal': 'blue', 'Anomaly': 'red'}, data=df)
    plt.title(title)
    plt.legend(title=label_col, loc='upper right')
    plt.show()

# Visualize results
for model_name in models.keys():
    plot_results(tsne_df, 't-SNE1', 't-SNE2', model_name, f"{model_name} (t-SNE)")

