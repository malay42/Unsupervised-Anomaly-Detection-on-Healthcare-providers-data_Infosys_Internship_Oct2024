import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score, pairwise_distances
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV

# Load and scale data
df = pd.read_csv('data_encoded.csv')
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

# Apply Isolation Forest
isolation_forest = IsolationForest(contamination=0.05, random_state=42)
labels = isolation_forest.fit_predict(data_scaled)
df['if_anomaly'] = labels
df['ifb_anomaly'] = np.where(df['if_anomaly'] == -1, 1, 0)

# PCA transformation for clustering quality
pca = PCA(n_components=2)
pca_df = pca.fit_transform(df.drop(['ifb_anomaly', 'if_anomaly'], axis=1))
score = calinski_harabasz_score(pca_df, df['ifb_anomaly'])
print(f"Calinski-Harabasz Score: {score}")

# Define hyperparameters for tuning
param_dist = {
    'n_estimators': [50, 100, 150],
    'max_samples': [0.6, 0.8, 1.0],
    'contamination': [0.04, 0.05, 0.06],
    'max_features': [0.5, 0.7, 1.0]
}

# Custom score function for tuning IsolationForest using Calinski-Harabasz score
def custom_score(estimator, X):
    labels = estimator.fit_predict(X)
    pca = PCA(n_components=2)
    pca_df = pca.fit_transform(X)
    return calinski_harabasz_score(pca_df, labels)

# Randomized Search with custom scoring function
random_search = RandomizedSearchCV(
    estimator=isolation_forest,
    param_distributions=param_dist,
    scoring=custom_score,
    cv=3,
    n_iter=10,
    random_state=42
)
random_search.fit(data_scaled)
best_params = random_search.best_params_
print("Best Parameters from Randomized Search:", best_params)

# Bayesian Optimization using BayesSearchCV
bayes_search = BayesSearchCV(
    estimator=isolation_forest,
    search_spaces=param_dist,
    scoring=custom_score,
    n_iter=10,
    cv=3,
    random_state=42
)
bayes_search.fit(data_scaled)
best_params_bayes = bayes_search.best_params_
print("Best Parameters from Bayesian Search:", best_params_bayes)

# Custom score function for calculating intra-cluster distance
def custom_score2(estimator, X, y=None):
    labels = estimator.fit_predict(X)
    normal_data = X[labels == 1]
    centroid = normal_data.mean(axis=0)
    distances = pairwise_distances(normal_data, [centroid])
    return -distances.mean()  # We want to minimize the distance

# Randomized Search with custom intra-cluster distance scoring
random_search2 = RandomizedSearchCV(
    estimator=isolation_forest,
    param_distributions=param_dist,
    scoring=custom_score2,
    cv=3,
    n_iter=10,
    random_state=42
)
random_search2.fit(data_scaled)
best_params2 = random_search2.best_params_
print("Best Parameters from Randomized Search (Intra-cluster Distance):", best_params2)

# Bayesian Optimization using BayesSearchCV for intra-cluster distance
bayes_search2 = BayesSearchCV(
    estimator=isolation_forest,
    search_spaces=param_dist,
    scoring=custom_score2,
    n_iter=10,
    cv=3,
    random_state=42
)
bayes_search2.fit(data_scaled)
best_params_bayes2 = bayes_search2.best_params_
print("Best Parameters from Bayesian Search (Intra-cluster Distance):", best_params_bayes2)
