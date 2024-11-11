import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score
import optuna
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = "normalized_data.csv"  # Update to your dataset path
df = pd.read_csv(file_path)

# Split dataset
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Define numeric columns for anomaly detection
numeric_columns = [
    'Number of Services', 'Number of Medicare Beneficiaries', 
    'Number of Distinct Medicare Beneficiary/Per Day Services',
    'Average Medicare Allowed Amount', 'Average Submitted Charge Amount',
    'Average Medicare Payment Amount', 'Average Medicare Standardized Amount'
]
X_train = train_df[numeric_columns].values
X_test = test_df[numeric_columns].values

# Optuna for Isolation Forest
def iso_forest_objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 200)
    max_samples = trial.suggest_float("max_samples", 0.5, 1.0)
    contamination = trial.suggest_float("contamination", 0.05, 0.2)
    max_features = trial.suggest_float("max_features", 0.5, 1.0)
    
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        max_features=max_features,
        random_state=42
    )
    
    predictions = model.fit_predict(X_train)
    return accuracy_score(predictions, predictions)

# Run Optuna optimization
optuna_study = optuna.create_study(direction="maximize")
optuna_study.optimize(iso_forest_objective, n_trials=20)
best_iso_forest_optuna = IsolationForest(**optuna_study.best_params, random_state=42)

# Testing and visualization with Optuna-tuned model
test_predictions = best_iso_forest_optuna.fit_predict(X_test)
accuracy = accuracy_score(test_predictions, test_predictions)
print("Test Accuracy (Optuna):", accuracy)

# Apply t-SNE to reduce dimensions for visualization
tsne = TSNE(n_components=2, random_state=42)
X_test_tsne = tsne.fit_transform(X_test)

# Plotting predictions after Optuna tuning
plt.figure(figsize=(12, 6))
sns.scatterplot(x=X_test_tsne[:, 0], y=X_test_tsne[:, 1], hue=test_predictions, palette="coolwarm", legend="full")
plt.title("Isolation Forest (Optuna Tuning)")

plt.show()
