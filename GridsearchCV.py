import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('cleaned2_encoded.csv')

# Preprocess the data
data = data.select_dtypes(include=['float', 'int'])

# Define the parameter grid
param_grid = {
    'n_estimators': [40, 50, 60],
    'max_samples': [0.9, 1.0],
    'max_features': [0.9, 1.0],
    'contamination': [0.04, 0.05, 0.06]
}

# Define a custom scoring function
def isolation_forest_score(estimator, X):
    return -estimator.score_samples(X).mean()

# Create the Isolation Forest model
model = IsolationForest()

# Perform grid search with custom scoring
grid_search = GridSearchCV(model, param_grid, cv=5, scoring=isolation_forest_score)
grid_search.fit(data)

# Print the best parameters and score
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_}")
