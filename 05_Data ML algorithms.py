# Isolation Forest method

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load your data
data = pd.read_csv("encoded.csv")

# Select features for Isolation Forest
features = [
    'Number of Services', 'Number of Medicare Beneficiaries', 
    'Number of Distinct Medicare Beneficiary/Per Day Services', 
    'Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 
    'Average Medicare Payment Amount', 'Average Medicare Standardized Amount'
]

# Scaling the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features])

# Initialize and fit Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42, n_estimators=5)
iso_forest.fit(data_scaled)

# Predict anomalies
data['anomaly'] = iso_forest.predict(data_scaled)

# Separate normal and anomaly data
normal_data = data[data['anomaly'] == 1]
anomalies = data[data['anomaly'] == -1]

# Calculate total number of anomalies
total_anomalies = len(anomalies)
print(f"Total number of anomalies detected: {total_anomalies}")


# Define the percentiles you want to use for scaling
x_min, x_max = normal_data['Number of Services'].quantile(0.01), normal_data['Number of Services'].quantile(0.99)
y_min, y_max = normal_data['Average Medicare Payment Amount'].quantile(0.01), normal_data['Average Medicare Payment Amount'].quantile(0.99)

# Plot the anomalies and normal data
plt.figure(figsize=(10, 6))

# Choose two features to plot; here we use 'Number of Services' and 'Average Medicare Payment Amount'
plt.scatter(normal_data['Number of Services'], normal_data['Average Medicare Payment Amount'], 
            c='blue', label='Normal', alpha=0.6)
plt.scatter(anomalies['Number of Services'], anomalies['Average Medicare Payment Amount'], 
            c='red', label='Anomaly', alpha=0.6)

# Adding title and labels
plt.title('Isolation Forest Anomalies')
plt.xlabel('Number of Services')
plt.ylabel('Average Medicare Payment Amount')
# Adjust axis limits
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend()
plt.show()

# DBSCAN method

import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load your data
data = pd.read_csv("encoded.csv")

# Select a subset of features to reduce dimensionality
features = ['Number of Services', 'Average Medicare Payment Amount']

# Sample a fraction of the data to reduce memory usage
sampled_data = data.sample(frac=0.1, random_state=42)

# Scale the sampled data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(sampled_data[features])

# Initialize and fit DBSCAN with smaller `eps` and higher `min_samples` values
dbscan = DBSCAN(eps=0.5, min_samples=10)
sampled_data['anomaly'] = dbscan.fit_predict(data_scaled)

# DBSCAN labels -1 as noise points (anomalies)
sampled_data['anomaly'] = sampled_data['anomaly'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')

# Separate normal and anomaly data
normal_data = sampled_data[sampled_data['anomaly'] == 'Normal']
anomalies = sampled_data[sampled_data['anomaly'] == 'Anomaly']

# Calculate total number of anomalies
total_anomalies = len(anomalies)
print(f"Total number of anomalies detected: {total_anomalies}")

# Define axis limits based on percentile to show more anomalies and clusters
x_min, x_max = normal_data['Number of Services'].quantile(0.01), normal_data['Number of Services'].quantile(0.99)
y_min, y_max = normal_data['Average Medicare Payment Amount'].quantile(0.01), normal_data['Average Medicare Payment Amount'].quantile(0.99)

# Plot the anomalies and normal data
plt.figure(figsize=(10, 6))

# Scatter plot with lower opacity for normal data
plt.scatter(normal_data['Number of Services'], normal_data['Average Medicare Payment Amount'], 
            c='blue', label='Normal', s=10, alpha=0.4)
plt.scatter(anomalies['Number of Services'], anomalies['Average Medicare Payment Amount'], 
            c='red', label='Anomaly', marker='x', s=50, alpha=0.7)

# Adding title and labels
plt.title('DBSCAN Anomalies')
plt.xlabel('Number of Services')
plt.ylabel('Average Medicare Payment Amount')

# Set custom axis limits based on percentile values
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.legend()
plt.show()


# DBSCAN anomalies graph

import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load your data
data = pd.read_csv("encoded.csv")

# Select a subset of features to reduce dimensionality
features = ['Number of Services', 'Average Medicare Payment Amount']

# Sample a fraction of the data to reduce memory usage
sampled_data = data.sample(frac=0.1, random_state=42)

# Scale the sampled data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(sampled_data[features])

# Initialize and fit DBSCAN with smaller `eps` and higher `min_samples` values
dbscan = DBSCAN(eps=0.5, min_samples=10)
sampled_data['anomaly'] = dbscan.fit_predict(data_scaled)

# DBSCAN labels -1 as noise points (anomalies)
sampled_data['anomaly'] = sampled_data['anomaly'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')

# Separate normal and anomaly data
normal_data = sampled_data[sampled_data['anomaly'] == 'Normal']
anomalies = sampled_data[sampled_data['anomaly'] == 'Anomaly']

# Calculate total number of anomalies
total_anomalies = len(anomalies)
print(f"Total number of anomalies detected: {total_anomalies}")

# Plot the anomalies and normal data
plt.figure(figsize=(10, 6))

# Scatter plot with lower opacity for normal data
plt.scatter(normal_data['Number of Services'], normal_data['Average Medicare Payment Amount'], 
            c='blue', label='Normal', s=10)
plt.scatter(anomalies['Number of Services'], anomalies['Average Medicare Payment Amount'], 
            c='red', label='Anomaly', marker='x', s=50)

# Adding title and labels
plt.title('DBSCAN Anomalies')
plt.xlabel('Number of Services')
plt.ylabel('Average Medicare Payment Amount')

# Adjust axis limits to automatic scaling to show all anomalies
plt.legend()
plt.show()


# KNN method 

import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load your data
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

# Initialize and fit Local Outlier Factor (KNN-based)
knn = LocalOutlierFactor(n_neighbors=20)  # Adjust n_neighbors as necessary
data['anomaly'] = knn.fit_predict(data_scaled)

# Separate normal and anomaly data
normal_data = data[data['anomaly'] == 1]
anomalies = data[data['anomaly'] == -1]

# Calculate total number of anomalies
total_anomalies = len(anomalies)
print(f"Total number of anomalies detected: {total_anomalies}")

# Define the percentiles you want to use for scaling
x_min, x_max = normal_data['Number of Services'].quantile(0.01), normal_data['Number of Services'].quantile(0.99)
y_min, y_max = normal_data['Average Medicare Payment Amount'].quantile(0.01), normal_data['Average Medicare Payment Amount'].quantile(0.99)

# Plot the anomalies and normal data
plt.figure(figsize=(10, 6))

# Choose two features to plot; here we use 'Number of Services' and 'Average Medicare Payment Amount'
plt.scatter(normal_data['Number of Services'], normal_data['Average Medicare Payment Amount'], 
            c='blue', label='Normal', alpha=0.6)
plt.scatter(anomalies['Number of Services'], anomalies['Average Medicare Payment Amount'], 
            c='red', label='Anomaly', alpha=0.6)

# Adding title and labels
plt.title('KNN Anomalies')
plt.xlabel('Number of Services')
plt.ylabel('Average Medicare Payment Amount')
# Adjust axis limits
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend()
plt.show()

# One-Class SVM 

import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load your data
data = pd.read_csv("encoded.csv")

# Select features for One-Class SVM
features = [
    'Number of Services', 'Number of Medicare Beneficiaries', 
    'Number of Distinct Medicare Beneficiary/Per Day Services', 
    'Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 
    'Average Medicare Payment Amount', 'Average Medicare Standardized Amount'
]

# Sample a fraction of the data for faster processing (e.g., 10%)
sampled_data = data.sample(frac=0.7, random_state=50)

# Scaling the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(sampled_data[features])

# Initialize and fit One-Class SVM
svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)  # Adjust nu and kernel as necessary
sampled_data['anomaly'] = svm.fit_predict(data_scaled)

# Separate normal and anomaly data
normal_data = sampled_data[sampled_data['anomaly'] == 1]
anomalies = sampled_data[sampled_data['anomaly'] == -1]

# Calculate total number of anomalies
total_anomalies = len(anomalies)
print(f"Total number of anomalies detected: {total_anomalies}")

# Define the percentiles you want to use for scaling
x_min, x_max = normal_data['Number of Services'].quantile(0.01), normal_data['Number of Services'].quantile(0.99)
y_min, y_max = normal_data['Average Medicare Payment Amount'].quantile(0.01), normal_data['Average Medicare Payment Amount'].quantile(0.99)

# Plot the anomalies and normal data
plt.figure(figsize=(10, 6))

# Choose two features to plot; here we use 'Number of Services' and 'Average Medicare Payment Amount'
plt.scatter(normal_data['Number of Services'], normal_data['Average Medicare Payment Amount'], 
            c='blue', label='Normal', alpha=0.6)
plt.scatter(anomalies['Number of Services'], anomalies['Average Medicare Payment Amount'], 
            c='red', label='Anomaly', alpha=0.6)

# Adding title and labels
plt.title('One-Class SVM Anomalies')
plt.xlabel('Number of Services')
plt.ylabel('Average Medicare Payment Amount')
# Adjust axis limits
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend()
plt.show()