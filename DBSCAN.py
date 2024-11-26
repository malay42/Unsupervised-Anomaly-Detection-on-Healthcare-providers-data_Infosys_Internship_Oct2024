import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# the dataset
df = pd.read_csv("encoded.csv")

# Few features for clustering
selected_features = df[['Number of Services', 'Average Medicare Payment Amount', 
                        'Average Submitted Charge Amount', 'Average Medicare Standardized Amount']]

# a portion of the data for efficiency
sample_fraction = 0.1
sampled_df = selected_features.sample(frac=sample_fraction, random_state=42)

# Scale the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(sampled_df)

# DBSCAN clustering
dbscan_model = DBSCAN(eps=0.5, min_samples=10)
cluster_labels = dbscan_model.fit_predict(scaled_data)

# Add cluster labels to the sampled data
sampled_df['Cluster'] = cluster_labels
sampled_df['Cluster'] = sampled_df['Cluster'].apply(lambda label: 'Anomaly' if label == -1 else 'Normal')

# Count anomalies
anomaly_count = (sampled_df['Cluster'] == 'Anomaly').sum()
print(f"Detected Anomalies: {anomaly_count}")

# axis limits based on quantiles of the primary features for visualization
x_range = sampled_df['Number of Services'].quantile([0.01, 0.99]).values
y_range = sampled_df['Average Medicare Payment Amount'].quantile([0.01, 0.99]).values

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(
    sampled_df[sampled_df['Cluster'] == 'Normal']['Number of Services'],
    sampled_df[sampled_df['Cluster'] == 'Normal']['Average Medicare Payment Amount'],
    c='blue', label='Normal', s=10, alpha=0.4
)
plt.scatter(
    sampled_df[sampled_df['Cluster'] == 'Anomaly']['Number of Services'],
    sampled_df[sampled_df['Cluster'] == 'Anomaly']['Average Medicare Payment Amount'],
    c='red', label='Anomaly', marker='x', s=50, alpha=0.7
)

# ploting
plt.title("DBSCAN Clustering with Additional Features")
plt.xlabel("Number of Services")
plt.ylabel("Average Medicare Payment Amount")
plt.xlim(x_range[0], x_range[1])
plt.ylim(y_range[0], y_range[1])
plt.legend()
plt.tight_layout()
plt.show()
