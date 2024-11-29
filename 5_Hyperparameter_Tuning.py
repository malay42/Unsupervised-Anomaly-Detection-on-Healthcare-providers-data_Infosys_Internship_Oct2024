# Importing Libraries
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, calinski_harabasz_score
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV

# Function Definitions
def calculate_intra_cluster_variance(data, labels):
    """Calculate average distance of normal points from the centroid."""
    normal_data = data[labels == 1]
    if normal_data.shape[0] == 0:
        return np.nan
    centroid = normal_data.mean(axis=0)
    distances = pairwise_distances(normal_data, [centroid])
    return distances.mean()

def custom_score(estimator, X, y=None):
    """Custom scoring function: negative intra-cluster variance."""
    labels = estimator.fit_predict(X)
    return -calculate_intra_cluster_variance(X, labels) 

def custom_score_ch(estimator, X, y=None):
    """Custom scoring function: Calinski-Harabasz score using PCA."""
    labels = estimator.fit_predict(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    return calinski_harabasz_score(X_pca, labels)

def perform_grid_search(data_scaled):
    """Perform Grid Search for Isolation Forest parameters."""
    contamination = [0.04, 0.05, 0.06]
    n_estimators = [50, 100, 150, 200]
    max_samples = [0.1, 0.5, 1.0]

    results = []
    best_params = None
    best_icv_score = float('inf')

    for cont in contamination:
        for n_est in n_estimators:
            for max_samp in max_samples:
                
                isolation_forest = IsolationForest(
                    contamination=cont,
                    n_estimators=n_est,
                    max_samples=max_samp,
                    random_state=42
                )
                labels = isolation_forest.fit_predict(data_scaled)

                icv_score = calculate_intra_cluster_variance(data_scaled, labels)
                if icv_score < best_icv_score:
                    best_icv_score = icv_score
                    best_params = {
                        'contamination': cont,
                        'n_estimators': n_est,
                        'max_samples': max_samp
                    }

    return best_params

def perform_randomized_search(data, model, parameters):
    """Perform Randomized Search for Isolation Forest parameters."""
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=parameters,
        scoring=custom_score,
        cv=3,
        n_iter=10,
        random_state=42
    )
    random_search.fit(data)
    random_search_params = random_search.best_params_
    
    return random_search_params

def perform_bayesian_search(data, model, parameters):
    """Perform Bayesian Search for Isolation Forest parameters."""
    bayes_search = BayesSearchCV(
        estimator=model,
        search_spaces=parameters,
        scoring=custom_score,
        n_iter=10,
        cv=3,
        random_state=42
    )
    bayes_search.fit(data)
    bayesian_search_params = bayes_search.best_params_
    return bayesian_search_params

def evaluate_model_ch_score(model_params, data_scaled):
    """Function to evaluate Calinski-Harabasz score for the best models."""
    isolation_forest = IsolationForest(**model_params, random_state=42)
    labels = isolation_forest.fit_predict(data_scaled)
    return custom_score_ch(isolation_forest, data_scaled)

# Main Workflow
if __name__ == "__main__":

    df = pd.read_csv('data_encoded.csv')
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)

    # 1: Grid Search
    grid_search_params = perform_grid_search(data_scaled)
    print("Best Parameters from Grid Search:", grid_search_params)
    
    param_dist = {
        'n_estimators': [50, 100, 150],
        'max_samples': [0.6, 0.8, 1.0],
        'contamination': [0.04, 0.05, 0.06],
        'max_features': [0.5, 0.7, 1.0]
    }
    isolation_forest = IsolationForest(random_state=42)

    # 2: Randomized Search
    random_search_params = perform_randomized_search(data_scaled, isolation_forest, param_dist)
    print("Best Parameters from Randomized Search:", random_search_params)

    # 3: Bayesian Search
    bayesian_search_params = perform_bayesian_search(data_scaled, isolation_forest, param_dist)
    print("Best Parameters from Bayesian Search:", bayesian_search_params)

    # Evaluate Calinski-Harabasz score for each model
    grid_ch_score = evaluate_model_ch_score(grid_search_params, data_scaled)
    print(f"Calinski-Harabasz Score from Grid Search: {grid_ch_score}")

    random_ch_score = evaluate_model_ch_score(random_search_params, data_scaled)
    print(f"Calinski-Harabasz Score from Randomized Search: {random_ch_score}")

    bayes_ch_score = evaluate_model_ch_score(bayesian_search_params, data_scaled)
    print(f"Calinski-Harabasz Score from Bayesian Search: {bayes_ch_score}")
