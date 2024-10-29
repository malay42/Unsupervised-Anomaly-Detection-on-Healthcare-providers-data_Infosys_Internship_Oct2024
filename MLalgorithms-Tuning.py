import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Generate synthetic data 
np.random.seed(0)
df = pd.read_csv("Cleaned_Healthcare Providers.csv")  # Replace with your file path
numerical_columns = ['Number of Services', 'Number of Medicare Beneficiaries', 
                     'Number of Distinct Medicare Beneficiary/Per Day Services',
                     'Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 
                     'Average Medicare Payment Amount', 'Average Medicare Standardized Amount']

data = df[numerical_columns].copy()

# Using IQR method
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

# Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(X_train_pca)
y_train_iso = iso_forest.predict(X_train_pca)
y_test_iso = iso_forest.predict(X_test_pca)

# OneClass SVM
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

# Calculate accuracy (example for Isolation Forest)
accuracy_train_iso = np.mean(y_train_iso == 1)
accuracy_test_iso = np.mean(y_test_iso == 1)
print(f'Isolation Forest - Train Accuracy: {accuracy_train_iso:.2f}')
print(f'Isolation Forest - Test Accuracy: {accuracy_test_iso:.2f}')
