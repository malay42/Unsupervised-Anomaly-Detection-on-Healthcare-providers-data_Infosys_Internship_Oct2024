# Hyperparameter Tuning

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import optuna
import seaborn as sns
import matplotlib.pyplot as plt

# Load the processed data
file_path = "cleaned_data.csv"
df = pd.read_csv(file_path)

# Check and handle missing values in the dataset
print("Missing values in dataset:\n", df.isnull().sum())
imputer = SimpleImputer(strategy="mean")
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Split dataset into train and test sets
X_train, X_test = train_test_split(df.values, test_size=0.2, random_state=42)

# Optuna optimization function
def iso_forest_objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 200)
    max_samples = trial.suggest_float("max_samples", 0.5, 1.0)
    contamination = trial.suggest_float("contamination", 0.05, 0.2)
    max_features = trial.suggest_float("max_features", 0.5, 1.0)

    # Create Isolation Forest model
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        max_features=max_features,
        random_state=42
    )
    model.fit(X_train)

    predictions = model.predict(X_train)
    predictions_binary = np.where(predictions == 1, 1, 0)
    return accuracy_score(np.ones_like(predictions_binary), predictions_binary)

# Run Optuna
optuna_study = optuna.create_study(direction="maximize")
optuna_study.optimize(iso_forest_objective, n_trials=20)
best_params = optuna_study.best_params

print("Best parameters from Optuna:", best_params)

# Train the Isolation Forest with optimal parameters
best_model = IsolationForest(**best_params, random_state=42)
best_model.fit(X_train)

# Test predictions
test_predictions = best_model.predict(X_test)
test_predictions_binary = np.where(test_predictions == 1, 1, 0)

# Evaluate metrics
accuracy = accuracy_score(np.ones_like(test_predictions_binary), test_predictions_binary)
precision = precision_score(np.ones_like(test_predictions_binary), test_predictions_binary)
recall = recall_score(np.ones_like(test_predictions_binary), test_predictions_binary)
f1 = f1_score(np.ones_like(test_predictions_binary), test_predictions_binary)

print(f"Test Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

# t-SNE Visualization
tsne = TSNE(n_components=2, random_state=42)
X_test_tsne = tsne.fit_transform(X_test)

plt.figure(figsize=(12, 6))
sns.scatterplot(
    x=X_test_tsne[:, 0],
    y=X_test_tsne[:, 1],
    hue=test_predictions_binary,
    palette={0: "red", 1: "green"},
    legend="full"
)
plt.title("Isolation Forest (Optuna-Tuned) - Anomaly Detection")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend(title="Prediction", labels=["Anomaly", "Normal"])
plt.show()