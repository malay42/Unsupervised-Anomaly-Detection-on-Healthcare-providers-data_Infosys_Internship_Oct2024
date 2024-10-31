import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from skopt import BayesSearchCV

df = pd.read_csv("Cleaned_Healthcare Providers.csv")
numerical_columns = ['Number of Services', 'Number of Medicare Beneficiaries',
                     'Number of Distinct Medicare Beneficiary/Per Day Services',
                     'Average Medicare Allowed Amount', 'Average Submitted Charge Amount',
                     'Average Medicare Payment Amount', 'Average Medicare Standardized Amount']
data = df[numerical_columns].copy()

# Anomaly detection and removal using IQR method
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

# Normalize and Standardize the dataset
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)
standardizer = StandardScaler()
standardized_data = standardizer.fit_transform(normalized_data)

# Train-test split (70-30)
X_train, X_test = train_test_split(standardized_data, test_size=0.3, random_state=42)

# Apply PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Initial Model Evaluation
# Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(X_train_pca)
y_train_iso = iso_forest.predict(X_train_pca)
y_test_iso = iso_forest.predict(X_test_pca)

# One-Class SVM
oc_svm = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')
oc_svm.fit(X_train_pca)
y_train_svm = oc_svm.predict(X_train_pca)
y_test_svm = oc_svm.predict(X_test_pca)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
y_train_dbscan = dbscan.fit_predict(X_train_pca)
y_test_dbscan = dbscan.fit_predict(X_test_pca)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_pca, np.zeros(len(X_train_pca)))  # Assuming normal data points
y_train_knn = knn.predict(X_train_pca)
y_test_knn = knn.predict(X_test_pca)

# Visualization
def plot_pca_results(X_pca, y, title):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis', s=50)
    plt.title(title)
    plt.show()

plot_pca_results(X_train_pca, y_train_iso, 'Isolation Forest - Train')
plot_pca_results(X_test_pca, y_test_iso, 'Isolation Forest - Test')
plot_pca_results(X_train_pca, y_train_svm, 'OneClass SVM - Train')
plot_pca_results(X_test_pca, y_test_svm, 'OneClass SVM - Test')
plot_pca_results(X_train_pca, y_train_dbscan, 'DBSCAN - Train')
plot_pca_results(X_test_pca, y_test_dbscan, 'DBSCAN - Test')
plot_pca_results(X_train_pca, y_train_knn, 'KNN - Train')
plot_pca_results(X_test_pca, y_test_knn, 'KNN - Test')

# Function to evaluate the model
# Function to evaluate the model
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=1)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=1)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=1)
    return accuracy, precision, recall, f1

# Assuming normal data points as 1 for evaluation
y_true_train = np.ones(len(X_train_pca))
y_true_test = np.ones(len(X_test_pca))

# Evaluate Isolation Forest
accuracy_train_iso, precision_train_iso, recall_train_iso, f1_train_iso = evaluate_model(y_true_train, y_train_iso)
accuracy_test_iso, precision_test_iso, recall_test_iso, f1_test_iso = evaluate_model(y_true_test, y_test_iso)
print(f'Isolation Forest - Train Accuracy: {accuracy_train_iso:.2f}, Precision: {precision_train_iso:.2f}, Recall: {recall_train_iso:.2f}, F1 Score: {f1_train_iso:.2f}')
print(f'Isolation Forest - Test Accuracy: {accuracy_test_iso:.2f}, Precision: {precision_test_iso:.2f}, Recall: {recall_test_iso:.2f}, F1 Score: {f1_test_iso:.2f}')

# Evaluate One-Class SVM
accuracy_train_svm, precision_train_svm, recall_train_svm, f1_train_svm = evaluate_model(y_true_train, y_train_svm)
accuracy_test_svm, precision_test_svm, recall_test_svm, f1_test_svm = evaluate_model(y_true_test, y_test_svm)
print(f'One-Class SVM - Train Accuracy: {accuracy_train_svm:.2f}, Precision: {precision_train_svm:.2f}, Recall: {recall_train_svm:.2f}, F1 Score: {f1_train_svm:.2f}')
print(f'One-Class SVM - Test Accuracy: {accuracy_test_svm:.2f}, Precision: {precision_test_svm:.2f}, Recall: {recall_test_svm:.2f}, F1 Score: {f1_test_svm:.2f}')

# Evaluate DBSCAN
silhouette_train_dbscan = silhouette_score(X_train_pca, y_train_dbscan)
silhouette_test_dbscan = silhouette_score(X_test_pca, y_test_dbscan)
print(f'DBSCAN - Train Silhouette Score: {silhouette_train_dbscan:.2f}')
print(f'DBSCAN - Test Silhouette Score: {silhouette_test_dbscan:.2f}')

# Evaluate KNN
accuracy_train_knn, precision_train_knn, recall_train_knn, f1_train_knn = evaluate_model(y_true_train, y_train_knn)
accuracy_test_knn, precision_test_knn, recall_test_knn, f1_test_knn = evaluate_model(y_true_test, y_test_knn)
print(f'KNN - Train Accuracy: {accuracy_train_knn:.2f}, Precision: {precision_train_knn:.2f}, Recall: {recall_train_knn:.2f}, F1 Score: {f1_train_knn:.2f}')
print(f'KNN - Test Accuracy: {accuracy_test_knn:.2f}, Precision: {precision_test_knn:.2f}, Recall: {recall_test_knn:.2f}, F1 Score: {f1_test_knn:.2f}')

# Define the parameter grid for Bayesian Optimization
param_grid = {
    'contamination': (0.01, 0.1, 'uniform'),
    'n_estimators': (50, 200)
}

# Perform Bayesian Optimization
opt = BayesSearchCV(iso_forest, param_grid, n_iter=20, cv=3, scoring='accuracy', random_state=42)
opt.fit(X_train_pca, np.ones(len(X_train_pca)))

# Get the best model after optimization
best_iso_forest = opt.best_estimator_

# Predictions
y_train_iso_opt = best_iso_forest.predict(X_train_pca)
y_test_iso_opt = best_iso_forest.predict(X_test_pca)

# Function to evaluate the model
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1)
    recall = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    return accuracy, precision, recall, f1

# Assuming normal data points as 1 for evaluation
y_true_train = np.ones(len(X_train_pca))
y_true_test = np.ones(len(X_test_pca))

# Evaluate the optimized Isolation Forest model
accuracy_train_iso, precision_train_iso, recall_train_iso, f1_train_iso = evaluate_model(y_true_train, y_train_iso_opt)
accuracy_test_iso, precision_test_iso, recall_test_iso, f1_test_iso = evaluate_model(y_true_test, y_test_iso_opt)

print(f'Optimized Isolation Forest - Train Accuracy: {accuracy_train_iso:.2f}, Precision: {precision_train_iso:.2f}, Recall: {recall_train_iso:.2f}, F1 Score: {f1_train_iso:.2f}')
print(f'Optimized Isolation Forest - Test Accuracy: {accuracy_test_iso:.2f}, Precision: {precision_test_iso:.2f}, Recall: {recall_test_iso:.2f}, F1 Score: {f1_test_iso:.2f}')

# Visualization function
def plot_pca_results(X_pca, y, title):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis', s=50)
    plt.title(title)
    plt.show()

# Plot results for optimized Isolation Forest
plot_pca_results(X_train_pca, y_train_iso_opt, 'Optimized Isolation Forest - Train')
plot_pca_results(X_test_pca, y_test_iso_opt, 'Optimized Isolation Forest - Test')