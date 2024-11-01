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
import optuna
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


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

# ---------- Function to Evaluate and Plot Results ----------
def evaluate_and_plot(results, prefix=""):
    scores = {}
    anomaly_counts = {}

    for method, labels in results.items():
        score = silhouette_score(scaled_sampled_df, labels) if len(set(labels)) > 1 else np.nan
        scores[method] = score
        anomaly_counts[method] = np.sum(labels == -1)  # Count anomalies
        print(f"{prefix}{method} Silhouette Score: {score:.2f}, Anomalies Detected: {anomaly_counts[method]}")

    # Plotting silhouette scores
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(scores.keys()), y=list(scores.values()), palette='viridis')
    plt.title(f'Silhouette Scores for Each Anomaly Detection Method {prefix}')
    plt.ylabel('Silhouette Score')
    plt.xticks(rotation=45)
    plt.show()

    # Plotting the number of anomalies detected
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(anomaly_counts.keys()), y=list(anomaly_counts.values()), palette='pastel')
    plt.title(f'Number of Anomalies Detected by Each Method {prefix}')
    plt.ylabel('Number of Anomalies Detected')
    plt.xticks(rotation=45)
    plt.show()

# ---------- Anomaly Detection (Manual Hyperparameter Tuning) ----------
manual_results = {}

# Isolation Forest
manual_iso_forest = IsolationForest(n_estimators=200, contamination=0.05, max_samples=0.8, random_state=42)
manual_results['Isolation Forest'] = manual_iso_forest.fit_predict(scaled_sampled_df)

# One-Class SVM
manual_svm_model = OneClassSVM(kernel='rbf', gamma=0.005, nu=0.02)
manual_results['One-Class SVM'] = manual_svm_model.fit_predict(scaled_sampled_df)

# Local Outlier Factor
manual_lof = LocalOutlierFactor(n_neighbors=15, contamination=0.05)
manual_results['Local Outlier Factor'] = manual_lof.fit_predict(scaled_sampled_df)

# DBSCAN
manual_dbscan = DBSCAN(eps=1.2, min_samples=10)
manual_results['DBSCAN'] = manual_dbscan.fit_predict(scaled_sampled_df)

# Evaluate and plot results after manual tuning
evaluate_and_plot(manual_results, prefix="Manual Tuning: ")

# ---------- Hyperparameter Tuning with Optuna ----------
optuna_results = {}

def objective_iso(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    contamination = trial.suggest_float('contamination', 0.01, 0.1)
    iso_forest = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
    labels = iso_forest.fit_predict(scaled_sampled_df)
    return silhouette_score(scaled_sampled_df, labels) if len(set(labels)) > 1 else -1

study_iso = optuna.create_study(direction='maximize')
study_iso.optimize(objective_iso, n_trials=50)
best_iso_params = study_iso.best_params
print("Best params for Isolation Forest:", best_iso_params)

# Final Isolation Forest with best params
final_iso_forest = IsolationForest(**best_iso_params)
optuna_results['Isolation Forest'] = final_iso_forest.fit_predict(scaled_sampled_df)

def objective_svm(params):
    model = OneClassSVM(**params)
    labels = model.fit_predict(scaled_sampled_df)
    return {'loss': -silhouette_score(scaled_sampled_df, labels), 'status': STATUS_OK}

space_svm = {
    'kernel': hp.choice('kernel', ['linear', 'rbf', 'poly']),
    'gamma': hp.uniform('gamma', 0.001, 0.1),
    'nu': hp.uniform('nu', 0.01, 0.1)
}

best_svm = fmin(objective_svm, space_svm, algo=tpe.suggest, max_evals=50)
print("Best params for One-Class SVM:", best_svm)

# Final One-Class SVM with best params
final_svm_model = OneClassSVM(kernel=['linear', 'rbf', 'poly'][best_svm['kernel']], 
                               gamma=best_svm['gamma'], nu=best_svm['nu'])
optuna_results['One-Class SVM'] = final_svm_model.fit_predict(scaled_sampled_df)

lof_space = {
    'n_neighbors': hp.randint('n_neighbors', 5, 30),
    'contamination': hp.uniform('contamination', 0.01, 0.1)
}

def objective_lof(params):
    model = LocalOutlierFactor(**params)
    labels = model.fit_predict(scaled_sampled_df)
    return {'loss': -silhouette_score(scaled_sampled_df, labels), 'status': STATUS_OK}

best_lof = fmin(objective_lof, lof_space, algo=tpe.suggest, max_evals=50)
print("Best params for Local Outlier Factor:", best_lof)

# Final Local Outlier Factor with best params
final_lof = LocalOutlierFactor(n_neighbors=best_lof['n_neighbors'], contamination=best_lof['contamination'])
optuna_results['Local Outlier Factor'] = final_lof.fit_predict(scaled_sampled_df)

# DBSCAN Hyperparameter tuning using Optuna
def objective_dbscan(trial):
    eps = trial.suggest_float('eps', 0.1, 1.5)
    min_samples = trial.suggest_int('min_samples', 1, 30)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(scaled_sampled_df)
    return silhouette_score(scaled_sampled_df, labels) if len(set(labels)) > 1 else -1

study_dbscan = optuna.create_study(direction='maximize')
study_dbscan.optimize(objective_dbscan, n_trials=50)
best_dbscan_params = study_dbscan.best_params
print("Best params for DBSCAN:", best_dbscan_params)

# Final DBSCAN with best params
final_dbscan = DBSCAN(**best_dbscan_params)
optuna_results['DBSCAN'] = final_dbscan.fit_predict(scaled_sampled_df)

# Evaluate and plot results after Optuna tuning
evaluate_and_plot(optuna_results, prefix="Optuna Tuning: ")

# ---------- Hyperparameter Tuning with Hyperopt ----------
hyperopt_results = {}

# Isolation Forest Hyperopt
def hyperopt_objective_iso(params):
    iso_forest = IsolationForest(**params)
    labels = iso_forest.fit_predict(scaled_sampled_df)
    return {'loss': -silhouette_score(scaled_sampled_df, labels), 'status': STATUS_OK}

hyperopt_space_iso = {
    'n_estimators': hp.randint('n_estimators', 50, 300),
    'contamination': hp.uniform('contamination', 0.01, 0.1),
    'max_samples': hp.uniform('max_samples', 0.5, 1.0)
}

best_hyperopt_iso = fmin(hyperopt_objective_iso, hyperopt_space_iso, algo=tpe.suggest, max_evals=50)
print("Best params for Isolation Forest (Hyperopt):", best_hyperopt_iso)

# Final Isolation Forest with Hyperopt params
final_hyperopt_iso = IsolationForest(**best_hyperopt_iso)
hyperopt_results['Isolation Forest'] = final_hyperopt_iso.fit_predict(scaled_sampled_df)

# One-Class SVM Hyperopt
hyperopt_space_svm = {
    'kernel': hp.choice('kernel', ['linear', 'rbf', 'poly']),
    'gamma': hp.uniform('gamma', 0.001, 0.1),
    'nu': hp.uniform('nu', 0.01, 0.1)
}

best_hyperopt_svm = fmin(objective_svm, hyperopt_space_svm, algo=tpe.suggest, max_evals=50)
print("Best params for One-Class SVM (Hyperopt):", best_hyperopt_svm)

# Final One-Class SVM with Hyperopt params
final_hyperopt_svm = OneClassSVM(kernel=['linear', 'rbf', 'poly'][best_hyperopt_svm['kernel']], 
                                   gamma=best_hyperopt_svm['gamma'], nu=best_hyperopt_svm['nu'])
hyperopt_results['One-Class SVM'] = final_hyperopt_svm.fit_predict(scaled_sampled_df)

# LOF Hyperparameter tuning using Hyperopt
def objective_lof(params):
    n_neighbors = params['n_neighbors']
    contamination = params['contamination']
    
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    labels = lof.fit_predict(scaled_sampled_df)
    
    # Check the number of unique labels (LOF labels -1 for outliers, 1 for inliers)
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # Exclude outliers

    # If only one cluster or all points are outliers, return a high loss
    if n_clusters < 1:  # Check if there are less than 2 inliers
        return {'loss': 1.0, 'status': STATUS_OK}  # High loss for invalid clustering
    
    # Calculate the silhouette score only if there are valid inliers
    try:
        score = silhouette_score(scaled_sampled_df[labels == 1], labels[labels == 1])  # Only inliers
    except ValueError as e:
        score = 1.0  # Assign high loss if silhouette_score cannot be computed
    
    return {'loss': -score, 'status': STATUS_OK}

# Hyperparameter space for LOF
hyperopt_space_lof = {
    'n_neighbors': hp.randint('n_neighbors', 1, 50),
    'contamination': hp.uniform('contamination', 0.01, 0.1)
}

# Optimize the parameters using Hyperopt
best_hyperopt_lof = fmin(objective_lof, hyperopt_space_lof, algo=tpe.suggest, max_evals=50)
print("Best params for LOF (Hyperopt):", best_hyperopt_lof)

# Final Local Outlier Factor with Hyperopt params
final_hyperopt_lof = LocalOutlierFactor(n_neighbors=best_hyperopt_lof['n_neighbors'], contamination=best_hyperopt_lof['contamination'])
hyperopt_results['Local Outlier Factor'] = final_hyperopt_lof.fit_predict(scaled_sampled_df)

# DBSCAN Hyperparameter tuning using Hyperopt
def objective_dbscan(params):
    eps = params['eps']
    min_samples = params['min_samples']
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(scaled_sampled_df)
    
    # Check the number of unique labels
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Exclude noise (-1)
    
    # If only one cluster or all points are noise, return a high loss
    if n_clusters <= 1:
        return {'loss': 1.0, 'status': STATUS_OK}  # or use np.inf
    
    return {'loss': -silhouette_score(scaled_sampled_df, labels), 'status': STATUS_OK}

hyperopt_space_dbscan = {
    'eps': hp.uniform('eps', 0.1, 1.5),
    'min_samples': hp.randint('min_samples', 1, 30)
}

best_hyperopt_dbscan = fmin(objective_dbscan, hyperopt_space_dbscan, algo=tpe.suggest, max_evals=50)
print("Best params for DBSCAN (Hyperopt):", best_hyperopt_dbscan)

# Final DBSCAN with Hyperopt params
final_hyperopt_dbscan = DBSCAN(eps=best_hyperopt_dbscan['eps'], min_samples=best_hyperopt_dbscan['min_samples'])
hyperopt_results['DBSCAN'] = final_hyperopt_dbscan.fit_predict(scaled_sampled_df)

# Evaluate and plot results after Hyperopt tuning
evaluate_and_plot(hyperopt_results, prefix="Hyperopt Tuning: ")

# Print the final summary for reference
print("\nFinal Summary of Anomaly Detection Results:")
print("Manual Results:", manual_results)
print("Optuna Results:", optuna_results)
print("Hyperopt Results:", hyperopt_results)

from scipy.stats import ttest_rel, wilcoxon
import pandas as pd

# Create a DataFrame to store the silhouette scores and anomaly counts from each method
comparison_df = pd.DataFrame({
    "Method": ["Isolation Forest", "One-Class SVM", "Local Outlier Factor", "DBSCAN"],
    "Manual Silhouette": [silhouette_score(scaled_sampled_df, manual_results[method]) for method in manual_results.keys()],
    "Optuna Silhouette": [silhouette_score(scaled_sampled_df, optuna_results[method]) for method in optuna_results.keys()],
    "Hyperopt Silhouette": [silhouette_score(scaled_sampled_df, hyperopt_results[method]) for method in hyperopt_results.keys()],
    "Manual Anomalies": [np.sum(manual_results[method] == -1) for method in manual_results.keys()],
    "Optuna Anomalies": [np.sum(optuna_results[method] == -1) for method in optuna_results.keys()],
    "Hyperopt Anomalies": [np.sum(hyperopt_results[method] == -1) for method in hyperopt_results.keys()]
})

# Pairwise comparison of Silhouette scores between tuning methods
print("Pairwise T-Test Results for Silhouette Scores:")
methods = ["Manual", "Optuna", "Hyperopt"]
for i in range(len(methods)):
    for j in range(i + 1, len(methods)):
        t_score, p_value = ttest_rel(comparison_df[f"{methods[i]} Silhouette"], comparison_df[f"{methods[j]} Silhouette"])
        print(f"{methods[i]} vs {methods[j]}: t-score={t_score:.2f}, p-value={p_value:.3f}")

# Pairwise comparison of anomaly counts between tuning methods
print("\nPairwise Wilcoxon Test Results for Anomaly Counts:")
for i in range(len(methods)):
    for j in range(i + 1, len(methods)):
        rank_score, p_value = wilcoxon(comparison_df[f"{methods[i]} Anomalies"], comparison_df[f"{methods[j]} Anomalies"])
        print(f"{methods[i]} vs {methods[j]}: rank-score={rank_score:.2f}, p-value={p_value:.3f}")

# Display the summary DataFrame
print("\nSummary of Pairwise Comparison Results:")
print(comparison_df)

import matplotlib.pyplot as plt
import seaborn as sns

# Set up the visualization style
sns.set(style="whitegrid")

# Silhouette Score Comparison: Bar Plot
plt.figure(figsize=(12, 6))
silhouette_data = pd.melt(
    comparison_df[["Method", "Manual Silhouette", "Optuna Silhouette", "Hyperopt Silhouette"]],
    id_vars="Method", var_name="Tuning Method", value_name="Silhouette Score"
)
sns.barplot(data=silhouette_data, x="Method", y="Silhouette Score", hue="Tuning Method", palette="viridis")
plt.title("Silhouette Score Comparison Across Tuning Methods")
plt.ylabel("Average Silhouette Score")
plt.xlabel("Anomaly Detection Method")
plt.legend(title="Tuning Method", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Anomaly Count Comparison: Bar Plot
plt.figure(figsize=(12, 6))
anomaly_data = pd.melt(
    comparison_df[["Method", "Manual Anomalies", "Optuna Anomalies", "Hyperopt Anomalies"]],
    id_vars="Method", var_name="Tuning Method", value_name="Anomaly Count"
)
sns.barplot(data=anomaly_data, x="Method", y="Anomaly Count", hue="Tuning Method", palette="pastel")
plt.title("Anomaly Count Comparison Across Tuning Methods")
plt.ylabel("Number of Anomalies Detected")
plt.xlabel("Anomaly Detection Method")
plt.legend(title="Tuning Method", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
