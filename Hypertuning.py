import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import optuna
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data (ensure path is correct for your environment)
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
X_train, X_test, y_train, y_test = train_test_split(standardized_data, np.ones(len(standardized_data)), test_size=0.3, random_state=42)

# Define custom scorer for GridSearchCV, RandomizedSearchCV, and Optuna
def custom_scorer(y_true, y_pred):
    y_pred = np.where(y_pred == -1, 1, 0)  # Convert -1 to 1 (anomaly), 1 to 0 (normal)
    return accuracy_score(y_true, y_pred)

# Custom scorer function for GridSearchCV and RandomizedSearchCV
accuracy_scorer = make_scorer(custom_scorer)

# ----------- 1. Grid Search Hyperparameter Tuning --------------
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_samples': [0.5, 0.8, 1.0],
    'contamination': [0.05, 0.1, 0.15],
}

grid_search = GridSearchCV(estimator=IsolationForest(random_state=42), param_grid=param_grid, 
                           scoring=accuracy_scorer, cv=5, n_jobs=-1)
grid_search.fit(X_train)
print("Best parameters from Grid Search: ", grid_search.best_params_)

# Predict and evaluate the best model from GridSearchCV
y_pred_grid = grid_search.best_estimator_.predict(X_test)
y_pred_grid = np.where(y_pred_grid == -1, 1, 0)  # Convert -1 to 1 (anomaly), 1 to 0 (normal)
print(f"Grid Search Accuracy: {accuracy_score(y_test, y_pred_grid)}")

# ----------- 2. Randomized Search Hyperparameter Tuning --------------
param_distributions = {
    'n_estimators': [50, 100, 150, 200, 300],
    'max_samples': [0.5, 0.8, 1.0, 'auto'],
    'contamination': [0.05, 0.1, 0.15],
}

random_search = RandomizedSearchCV(estimator=IsolationForest(random_state=42), param_distributions=param_distributions, 
                                   scoring=accuracy_scorer, n_iter=10, cv=5, n_jobs=-1, random_state=42)
random_search.fit(X_train)
print("Best parameters from Random Search: ", random_search.best_params_)

# Predict and evaluate the best model from RandomizedSearchCV
y_pred_random = random_search.best_estimator_.predict(X_test)
y_pred_random = np.where(y_pred_random == -1, 1, 0)  # Convert -1 to 1 (anomaly), 1 to 0 (normal)
print(f"Random Search Accuracy: {accuracy_score(y_test, y_pred_random)}")

# ----------- 3. Bayesian Optimization with Optuna -------------------
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300, step=50)
    max_samples = trial.suggest_categorical('max_samples', [0.5, 0.8, 1.0, 'auto'])
    contamination = trial.suggest_float('contamination', 0.05, 0.15, step=0.05)

    model = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, contamination=contamination, random_state=42)
    model.fit(X_train)
    
    y_pred = model.predict(X_test)
    y_pred = np.where(y_pred == -1, 1, 0)  # Convert -1 to 1 (anomaly), 1 to 0 (normal)

    # Return negative of accuracy as Optuna minimizes the objective
    return -accuracy_score(y_test, y_pred)

# Create an Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print("Best parameters from Optuna: ", study.best_params)
print(f"Optuna Best Accuracy: {-study.best_value}")

# ----------- Visualize Results ------------------------
# Combine results for visualization
results = pd.DataFrame({
    'Method': ['Grid Search', 'Random Search', 'Optuna'],
    'Accuracy': [
        accuracy_score(y_test, np.where(grid_search.best_estimator_.predict(X_test) == -1, 1, 0)),
        accuracy_score(y_test, np.where(random_search.best_estimator_.predict(X_test) == -1, 1, 0)),
        -study.best_value
    ]
})

# Plot the accuracies of different tuning methods
plt.figure(figsize=(8, 5))
sns.barplot(x='Method', y='Accuracy', data=results)
plt.title('Accuracy of Anomaly Detection Models after Hyperparameter Tuning')
plt.xlabel('Tuning Method')
plt.ylabel('Accuracy')
plt.show()