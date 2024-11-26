# Hyperparameter Tuning for Anomaly Detection

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import optuna
import matplotlib.pyplot as plt

# Load the dataset
data_file = "cleaned2_encoded.csv"
data = pd.read_csv(data_file)

# Handle missing values
print("Missing values in dataset:\n", data.isnull().sum())
imputer = SimpleImputer(strategy="mean")
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Split the dataset into training and testing sets
X_train, X_test = train_test_split(data_imputed.values, test_size=0.2, random_state=42)

# Objective function for Optuna
def optimize_isolation_forest(trial):
    # Define the hyperparameter search space
    n_estimators = trial.suggest_int("n_estimators", 50, 200)
    max_samples = trial.suggest_float("max_samples", 0.5, 1.0)
    contamination = trial.suggest_float("contamination", 0.05, 0.2)
    max_features = trial.suggest_float("max_features", 0.5, 1.0)

    # Train Isolation Forest
    isolation_forest = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        max_features=max_features,
        random_state=42
    )
    isolation_forest.fit(X_train)

    # Generate predictions and calculate accuracy
    train_predictions = isolation_forest.predict(X_train)
    binary_predictions = np.where(train_predictions == 1, 1, 0)
    return accuracy_score(np.ones_like(binary_predictions), binary_predictions)

# Run hyperparameter optimization
study = optuna.create_study(direction="maximize")
study.optimize(optimize_isolation_forest, n_trials=20)
best_hyperparams = study.best_params

print("Optimal parameters found by Optuna:", best_hyperparams)

# Train the Isolation Forest model with the optimal parameters
final_model = IsolationForest(**best_hyperparams, random_state=42)
final_model.fit(X_train)

# Test the model
test_preds = final_model.predict(X_test)
binary_test_preds = np.where(test_preds == 1, 1, 0)

# Evaluate the model
accuracy = accuracy_score(np.ones_like(binary_test_preds), binary_test_preds)
precision = precision_score(np.ones_like(binary_test_preds), binary_test_preds)
recall = recall_score(np.ones_like(binary_test_preds), binary_test_preds)
f1 = f1_score(np.ones_like(binary_test_preds), binary_test_preds)

print(f"Test Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

# Visualization: Histogram of Predictions
plt.figure(figsize=(8, 5))
plt.hist(binary_test_preds, bins=3, color="skyblue", edgecolor="black", alpha=0.7)
plt.title("Anomaly Detection - Predictions Distribution")
plt.xlabel("Prediction (0: Anomaly, 1: Normal)")
plt.ylabel("Frequency")
plt.xticks([0, 1])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
