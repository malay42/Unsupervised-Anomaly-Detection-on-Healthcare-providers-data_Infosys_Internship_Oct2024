import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("encoded.csv")

# Select features for KNN
features = [
    'Number of Services', 'Number of Medicare Beneficiaries', 
    'Number of Distinct Medicare Beneficiary/Per Day Services', 
    'Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 
    'Average Medicare Payment Amount', 'Average Medicare Standardized Amount'
]

# Scaling the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features])

# Initialize Local Outlier Factor (KNN-based)
knn = LocalOutlierFactor(n_neighbors=20)  # You can experiment with n_neighbors
data['anomaly'] = knn.fit_predict(data_scaled)

# Separate normal and anomaly data
normal_data = data[data['anomaly'] == 1]
anomalies = data[data['anomaly'] == -1]

# Calculate total number of anomalies
total_anomalies = len(anomalies)
print(f"Total number of anomalies detected: {total_anomalies}")

# Plot the anomalies and normal data 
plt.figure(figsize=(10, 6))
x_min, x_max = normal_data['Number of Services'].quantile(0.01), normal_data['Number of Services'].quantile(0.99)
y_min, y_max = normal_data['Average Medicare Payment Amount'].quantile(0.01), normal_data['Average Medicare Payment Amount'].quantile(0.99)

# Plot Normal Data
plt.scatter(normal_data['Number of Services'], normal_data['Average Medicare Payment Amount'], 
            c='blue', label='Normal', alpha=0.6)

# Plot Anomalies
plt.scatter(anomalies['Number of Services'], anomalies['Average Medicare Payment Amount'], 
            c='red', label='Anomaly', alpha=0.6)

# Adding title and labels
plt.title('KNN Anomalies')
plt.xlabel('Number of Services')
plt.ylabel('Average Medicare Payment Amount')

# Adjust axis limits
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# show plot
plt.legend()
plt.tight_layout()
plt.show()
