# GridSearch CV

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np

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

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_samples': [0.5, 0.7, 1.0],
    'contamination': [0.05, 0.1, 0.15]
}

# Custom scoring function to minimize the number of anomalies
def anomaly_score(estimator, X):
    predictions = estimator.predict(X)
    anomalies = np.sum(predictions == -1)  # Count anomalies (-1 is the label for anomalies)
    return -anomalies  # Use negative count to minimize with GridSearchCV

# Initialize IsolationForest and GridSearchCV
iso_forest = IsolationForest(random_state=42)
grid_search = GridSearchCV(
    iso_forest, 
    param_grid=param_grid, 
    scoring=anomaly_score, 
    cv=3  # Cross-validation folds
)

# Fit GridSearchCV to the scaled data
grid_search.fit(data_scaled)

# Extract the best estimator and parameters
best_iso_forest = grid_search.best_estimator_
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# Predict anomalies with the best model
data['anomaly'] = best_iso_forest.predict(data_scaled)

# Separate normal and anomaly data
normal_data = data[data['anomaly'] == 1]
anomalies = data[data['anomaly'] == -1]

# Calculate total anomalies detected
total_anomalies = len(anomalies)
print(f"Total number of anomalies detected: {total_anomalies}")

# Plot the total anomalies for different parameters
results_df = pd.DataFrame(grid_search.cv_results_['params'])
results_df['total_anomalies'] = -grid_search.cv_results_['mean_test_score']

# plt.figure(figsize=(10, 6))
# plt.scatter(results_df['n_estimators'], results_df['total_anomalies'], c='blue', label='Total Anomalies')
# plt.title('Anomaly Counts for Different Isolation Forest Hyperparameters')
# plt.xlabel('Number of Estimators')
# plt.ylabel('Total Anomalies')
# plt.legend()
# plt.grid()
# plt.show()

# Define the percentiles for scaling
x_min, x_max = normal_data['Number of Services'].quantile(0.01), normal_data['Number of Services'].quantile(0.99)
y_min, y_max = normal_data['Average Medicare Payment Amount'].quantile(0.01), normal_data['Average Medicare Payment Amount'].quantile(0.99)

# Plot the anomalies and normal data using the best model
plt.figure(figsize=(10, 6))
plt.scatter(normal_data['Number of Services'], normal_data['Average Medicare Payment Amount'], 
            c='blue', label='Normal', alpha=0.6)
plt.scatter(anomalies['Number of Services'], anomalies['Average Medicare Payment Amount'], 
            c='red', label='Anomaly', alpha=0.6)

# Adding title and labels
plt.title('Best Isolation Forest Anomalies')
plt.xlabel('Number of Services')
plt.ylabel('Average Medicare Payment Amount')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend()
plt.show()

# Maunal Search

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

# Initialize lists to store results
results = []

# Adjusted parameter grids for increased anomaly detection
n_estimators_list = [50, 150, 175]  # Increased the number of estimators
max_samples_list = [0.3, 0.5, 0.7]  # Decreased the maximum samples
contamination_list = [0.1, 0.15, 0.2]  # Increased contamination rate

# Loop through each combination of parameters
for n_estimators in n_estimators_list:
    for max_samples in max_samples_list:
        for contamination in contamination_list:
            # Initialize and fit Isolation Forest with the current parameters
            iso_forest = IsolationForest(
                n_estimators=n_estimators, 
                max_samples=max_samples, 
                contamination=contamination, 
                random_state=42
            )
            iso_forest.fit(data_scaled)

            # Predict anomalies
            data['anomaly'] = iso_forest.predict(data_scaled)

            # Count anomalies
            total_anomalies = len(data[data['anomaly'] == -1])
            results.append((n_estimators, max_samples, contamination, total_anomalies))

# Convert results to DataFrame for better analysis
results_df = pd.DataFrame(results, columns=['n_estimators', 'max_samples', 'contamination', 'total_anomalies'])

# Find the best parameters based on the minimum number of anomalies
best_params = results_df.loc[results_df['total_anomalies'].idxmin()]

print(f"Best parameters found: {best_params[['n_estimators', 'max_samples', 'contamination']].to_dict()}")
print(f"Total number of anomalies detected: {best_params['total_anomalies']}")

# Visualize the best model's anomalies
best_iso_forest = IsolationForest(
    n_estimators=int(best_params['n_estimators']), 
    max_samples=best_params['max_samples'], 
    contamination=best_params['contamination'], 
    random_state=42
)

best_iso_forest.fit(data_scaled)
data['anomaly'] = best_iso_forest.predict(data_scaled)

# Separate normal and anomaly data
normal_data = data[data['anomaly'] == 1]
anomalies = data[data['anomaly'] == -1]

# Define the percentiles for scaling
x_min, x_max = normal_data['Number of Services'].quantile(0.01), normal_data['Number of Services'].quantile(0.99)
y_min, y_max = normal_data['Average Medicare Payment Amount'].quantile(0.01), normal_data['Average Medicare Payment Amount'].quantile(0.99)

# Plot the anomalies and normal data using the best model
plt.figure(figsize=(10, 6))
plt.scatter(normal_data['Number of Services'], normal_data['Average Medicare Payment Amount'], 
            c='blue', label='Normal', alpha=0.6)
plt.scatter(anomalies['Number of Services'], anomalies['Average Medicare Payment Amount'], 
            c='red', label='Anomaly', alpha=0.6)

# Adding title and labels
plt.title('Best Isolation Forest Anomalies')
plt.xlabel('Number of Services')
plt.ylabel('Average Medicare Payment Amount')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend()
plt.show()

# RandomSearch CV 

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import numpy as np

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

# Define updated parameter distributions for RandomizedSearchCV with lower anomaly sensitivity
param_distributions = {
    'n_estimators': [100, 200, 300],           # More trees for model stability
    'max_samples': [0.8, 0.9, 1.0],            # Larger samples to reduce sensitivity to minor outliers
    'contamination': [0.05, 0.03, 0.04],       # Lower contamination to detect fewer anomalies
    'max_features': [0.8, 1.0]                 # Use all or most features for better context in predictions
}

# Initialize Isolation Forest
iso_forest = IsolationForest(random_state=42)

# Custom scoring function to maximize proportion of detected anomalies
def anomaly_score(estimator, X):
    predictions = estimator.predict(X)
    anomalies = np.sum(predictions == -1)
    return -anomalies  # Use negative anomalies for minimizing in RandomizedSearchCV

# Set up RandomizedSearchCV with custom scoring
random_search = RandomizedSearchCV(
    iso_forest, 
    param_distributions=param_distributions,
    n_iter=10,                                # Reduce for faster tuning
    scoring=anomaly_score,                    # Custom scoring function
    random_state=42,
    cv=3                                      # Cross-validation folds
)

# Fit RandomizedSearchCV to data
random_search.fit(data_scaled)

# Get the best parameters and model
best_iso_forest = random_search.best_estimator_
best_params = random_search.best_params_
print("Best Parameters:", best_params)

# Predict anomalies with the best model
data['anomaly'] = best_iso_forest.predict(data_scaled)

# Separate normal and anomaly data
normal_data = data[data['anomaly'] == 1]
anomalies = data[data['anomaly'] == -1]

# Calculate total anomalies detected
total_anomalies = len(anomalies)
print(f"Total number of anomalies detected: {total_anomalies}")

# Define the percentiles for scaling
x_min, x_max = normal_data['Number of Services'].quantile(0.01), normal_data['Number of Services'].quantile(0.99)
y_min, y_max = normal_data['Average Medicare Payment Amount'].quantile(0.01), normal_data['Average Medicare Payment Amount'].quantile(0.99)

# Plot the anomalies and normal data using the best model
plt.figure(figsize=(10, 6))
plt.scatter(normal_data['Number of Services'], normal_data['Average Medicare Payment Amount'], 
            c='blue', label='Normal', alpha=0.6)
plt.scatter(anomalies['Number of Services'], anomalies['Average Medicare Payment Amount'], 
            c='red', label='Anomaly', alpha=0.6)

# Adding title and labels
plt.title('Isolation Forest Anomalies with Optimized Parameters')
plt.xlabel('Number of Services')
plt.ylabel('Average Medicare Payment Amount')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend()
plt.show()