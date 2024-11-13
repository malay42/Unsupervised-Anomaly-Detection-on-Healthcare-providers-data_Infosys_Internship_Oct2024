import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report  # For performance evaluation

# Load the preprocessed dataset (already cleaned and encoded)
final_dataset = pd.read_csv('final_healthcare_encoded_data.csv')

# Ensure numeric columns are correctly formatted
numeric_columns = [
    'Zip Code of the Provider', 'Number of Services', 'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services', 'Average Medicare Allowed Amount',
    'Average Submitted Charge Amount', 'Average Medicare Payment Amount',
    'Average Medicare Standardized Amount'
]
# Apply K-Nearest Neighbors for anomaly detection
n_neighbors = 5
knn = NearestNeighbors(n_neighbors=n_neighbors)
knn.fit(final_dataset[numeric_columns])

# Calculate the average distance to the nearest neighbors
distances, _ = knn.kneighbors(final_dataset[numeric_columns])
avg_distance = distances.mean(axis=1)

# Set a threshold to identify anomalies
threshold = avg_distance.mean() + avg_distance.std()
final_dataset['anomaly_knn'] = (avg_distance > threshold).astype(int)

# Separate normal data and anomalies
anomalies_knn = final_dataset[final_dataset['anomaly_knn'] == 1]
normal_data_knn = final_dataset[final_dataset['anomaly_knn'] == 0]

# Display anomalies
print("Anomalies detected by KNN:")
print(anomalies_knn.head())
print(f"Total number of anomalies detected by KNN: {anomalies_knn.shape[0]}")

# If you have a ground truth column for anomalies (e.g., 'True Anomaly' column), you can evaluate the model
if 'True Anomaly' in final_dataset.columns:
    # Compare predicted anomalies with true anomalies
    print("Performance Evaluation (F1-Score, Precision, Recall):")
    print(classification_report(df['True Anomaly'], final_dataset['anomaly_knn']))

# Scatter plot of anomalies vs normal data for KNN
plt.figure(figsize=(10, 6))
plt.scatter(normal_data_knn['Number of Services'], normal_data_knn['Average Medicare Payment Amount'], 
            color='blue', label='Normal', alpha=0.5)
plt.scatter(anomalies_knn['Number of Services'], anomalies_knn['Average Medicare Payment Amount'], 
            color='green', label='Anomalies (KNN)', alpha=0.7)
plt.xlabel('Number of Services')
plt.ylabel('Average Medicare Payment Amount')
plt.title('Anomaly Detection with KNN')
plt.legend()
plt.grid()
plt.show()

# Save the final dataset with KNN anomaly labels
final_dataset.to_csv('knn_anomalies.csv', index=False)
