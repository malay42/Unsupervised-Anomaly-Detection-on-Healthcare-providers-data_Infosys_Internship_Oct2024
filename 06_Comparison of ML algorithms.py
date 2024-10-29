# Bar Plot

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load your data
data = pd.read_csv("encoded.csv")

# Select features for anomaly detection algorithms
features = [
    'Number of Services', 'Number of Medicare Beneficiaries', 
    'Number of Distinct Medicare Beneficiary/Per Day Services', 
    'Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 
    'Average Medicare Payment Amount', 'Average Medicare Standardized Amount'
]

# Initialize dictionary to store the results for each algorithm
results = {}

# 1. Isolation Forest and Local Outlier Factor on the full dataset
algorithms = {
    "Isolation Forest": IsolationForest(contamination=0.05, random_state=42),
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20)
}

for name, algorithm in algorithms.items():
    preds = algorithm.fit_predict(data[features])  # Fit and predict on full data
    data[name + '_anomaly'] = preds
    anomalies = np.sum(preds == -1)
    results[name] = {"preds": preds, "num_anomalies": anomalies}

# 2. One-Class SVM on a 70% sample of the data
sampled_data = data.sample(frac=0.7, random_state=50)
scaler = StandardScaler()
sampled_data_scaled = scaler.fit_transform(sampled_data[features])

oneclass_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
svm_preds = oneclass_svm.fit_predict(sampled_data_scaled)
sampled_data["One-Class SVM_anomaly"] = svm_preds
anomalies_svm = np.sum(svm_preds == -1)
results["One-Class SVM"] = {"preds": svm_preds, "num_anomalies": anomalies_svm}

# Visualization and Comparison
# Bar plot for anomaly counts
plt.figure(figsize=(8, 6))
plt.bar(results.keys(), [results[name]["num_anomalies"] for name in results], color=['blue', 'green', 'orange'])
plt.title("Number of Anomalies Detected by Each Algorithm")
plt.xlabel("Algorithm")
plt.ylabel("Number of Anomalies")
plt.show()

# Scatter plots for visual comparison
# Define axis limits based on percentiles for consistent range across plots
x_min, x_max = data['Number of Services'].quantile(0.01), data['Number of Services'].quantile(0.99)
y_min, y_max = data['Average Medicare Payment Amount'].quantile(0.01), data['Average Medicare Payment Amount'].quantile(0.99)

plt.figure(figsize=(18, 5))
for i, (name, result) in enumerate(results.items(), 1):
    plt.subplot(1, 3, i)
    
    if name == "One-Class SVM":
        plot_data = sampled_data  # Plot only sampled data for SVM
        normal_data = plot_data[plot_data["One-Class SVM_anomaly"] == 1]
        anomalies = plot_data[plot_data["One-Class SVM_anomaly"] == -1]
    else:
        plot_data = data
        normal_data = plot_data[result["preds"] == 1]
        anomalies = plot_data[result["preds"] == -1]

    plt.scatter(normal_data['Number of Services'], normal_data['Average Medicare Payment Amount'], 
                c='blue', label='Normal', alpha=0.6, s=10)
    plt.scatter(anomalies['Number of Services'], anomalies['Average Medicare Payment Amount'], 
                c='red', label='Anomaly', alpha=0.6, s=10)
    
    plt.title(f"{name} - Anomaly Detection")
    plt.xlabel("Number of Services")
    plt.ylabel("Average Medicare Payment Amount")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend()

plt.tight_layout()
plt.show()


# Comparison and difference in graphs

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load your data
data = pd.read_csv("encoded.csv")

# Select features for anomaly detection algorithms
features = [
    'Number of Services', 'Number of Medicare Beneficiaries', 
    'Number of Distinct Medicare Beneficiary/Per Day Services', 
    'Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 
    'Average Medicare Payment Amount', 'Average Medicare Standardized Amount'
]

# Initialize dictionary to store the results for each algorithm
results = {}

# 1. Isolation Forest and Local Outlier Factor on the full dataset
algorithms = {
    "Isolation Forest": IsolationForest(contamination=0.05, random_state=42),
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20)
}

for name, algorithm in algorithms.items():
    preds = algorithm.fit_predict(data[features])  # Fit and predict on full data
    data[name + '_anomaly'] = preds
    anomalies = np.sum(preds == -1)
    results[name] = {"preds": preds, "num_anomalies": anomalies}

# 2. One-Class SVM on a 70% sample of the data
sampled_data = data.sample(frac=0.7, random_state=50)
scaler = StandardScaler()
sampled_data_scaled = scaler.fit_transform(sampled_data[features])

oneclass_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
svm_preds = oneclass_svm.fit_predict(sampled_data_scaled)
sampled_data["One-Class SVM_anomaly"] = svm_preds
anomalies_svm = np.sum(svm_preds == -1)
results["One-Class SVM"] = {"preds": svm_preds, "num_anomalies": anomalies_svm}

# Visualization of pairwise comparisons
x_min, x_max = data['Number of Services'].quantile(0.01), data['Number of Services'].quantile(0.99)
y_min, y_max = data['Average Medicare Payment Amount'].quantile(0.01), data['Average Medicare Payment Amount'].quantile(0.99)

# Pairs to compare
pairs = [("Isolation Forest", "Local Outlier Factor"), 
         ("Isolation Forest", "One-Class SVM"), 
         ("Local Outlier Factor", "One-Class SVM")]

plt.figure(figsize=(18, 5))

for i, (algo1, algo2) in enumerate(pairs, 1):
    plt.subplot(1, 3, i)
    
    # Get data for each algorithm in the pair
    if algo2 == "One-Class SVM":
        plot_data1 = data
        plot_data2 = sampled_data
    else:
        plot_data1 = plot_data2 = data
    
    normal_data1 = plot_data1[results[algo1]["preds"] == 1]
    anomalies1 = plot_data1[results[algo1]["preds"] == -1]
    normal_data2 = plot_data2[results[algo2]["preds"] == 1]
    anomalies2 = plot_data2[results[algo2]["preds"] == -1]
    
    # Plot normal and anomaly points for each algorithm in the pair
    plt.scatter(normal_data1['Number of Services'], normal_data1['Average Medicare Payment Amount'], 
                c='blue', label=f'{algo1} Normal', alpha=0.4, s=10)
    plt.scatter(anomalies1['Number of Services'], anomalies1['Average Medicare Payment Amount'], 
                c='red', label=f'{algo1} Anomaly', alpha=0.4, s=10)
    
    plt.scatter(normal_data2['Number of Services'], normal_data2['Average Medicare Payment Amount'], 
                c='green', label=f'{algo2} Normal', alpha=0.4, s=10, marker='x')
    plt.scatter(anomalies2['Number of Services'], anomalies2['Average Medicare Payment Amount'], 
                c='purple', label=f'{algo2} Anomaly', alpha=0.4, s=10, marker='x')
    
    plt.title(f"{algo1} vs {algo2}")
    plt.xlabel("Number of Services")
    plt.ylabel("Average Medicare Payment Amount")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend()

plt.tight_layout()
plt.show()

# Heatmap

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data
data = pd.read_csv("encoded.csv")

# Select features for anomaly detection algorithms
features = [
    'Number of Services', 'Number of Medicare Beneficiaries', 
    'Number of Distinct Medicare Beneficiary/Per Day Services', 
    'Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 
    'Average Medicare Payment Amount', 'Average Medicare Standardized Amount'
]

# Initialize dictionary to store the results for each algorithm
results = {}

# 1. Isolation Forest and Local Outlier Factor on the full dataset
algorithms = {
    "Isolation Forest": IsolationForest(contamination=0.05, random_state=42),
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20)
}

for name, algorithm in algorithms.items():
    preds = algorithm.fit_predict(data[features])  # Fit and predict on full data
    data[name + '_anomaly'] = preds
    results[name] = preds

# 2. One-Class SVM on a 70% sample of the data
sampled_data = data.sample(frac=0.7, random_state=50)
scaler = StandardScaler()
sampled_data_scaled = scaler.fit_transform(sampled_data[features])

oneclass_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
svm_preds = oneclass_svm.fit_predict(sampled_data_scaled)
sampled_data["One-Class SVM_anomaly"] = svm_preds

# Assign results to match indices for heatmap calculation
data.loc[sampled_data.index, "One-Class SVM_anomaly"] = svm_preds
results["One-Class SVM"] = data["One-Class SVM_anomaly"].fillna(0)  # Fill non-sampled data with 0 (neutral label)

# Calculate pairwise agreement for heatmap
agreement_matrix = np.zeros((len(results), len(results)))

algorithm_names = list(results.keys())
for i, algo1 in enumerate(algorithm_names):
    for j, algo2 in enumerate(algorithm_names):
        # Agreement ratio: how often both algorithms label the same data point as normal or anomaly
        agreement = np.mean(results[algo1] == results[algo2])
        agreement_matrix[i, j] = agreement

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(agreement_matrix, annot=True, xticklabels=algorithm_names, yticklabels=algorithm_names, cmap="coolwarm")
plt.title("Heatmap between Anomaly Detection Algorithms")
plt.xlabel("Algorithm")
plt.ylabel("Algorithm")
plt.show()
