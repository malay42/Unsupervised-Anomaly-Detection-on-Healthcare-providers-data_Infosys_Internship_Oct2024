from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import silhouette_score
import optuna

df = pd.read_csv('Healthcare_Providers.csv')        # Loading dataset

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Isolation Forest
iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
iso_forest.fit(X_scaled)
# Predict anomalies (-1 indicates an anomaly, 1 indicates normal)
iso_preds = iso_forest.predict(X_scaled)

# One-Class SVM
oc_svm = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.01)
oc_svm.fit(X_scaled)
svm_preds = oc_svm.predict(X_scaled)

# DBSCAN 
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X_scaled)
dbscan_preds = dbscan.labels_

# Collect model predictions
results_df = pd.DataFrame({
    'Isolation Forest': iso_preds,
    'One-Class SVM': svm_preds,
    'Dbscan' : dbscan_preds
})

# Calculate number of anomalies for each model
anomaly_counts = results_df.apply(lambda col: (col == -1).sum())
print("Anomalies detected by each model:\n", anomaly_counts)

# Check consistency between models (e.g., correlation of predictions)
print("Correlation between model predictions:\n", results_df.corr())

# Reduce dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plotting function
def plot_anomalies(X, labels, title):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette='coolwarm', s=20, marker='o')
    plt.title(f'Anomaly Detection with {title}')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(['Normal', 'Anomaly'])
    plt.show()

# Visualize each model's anomaly predictions
for model_name, labels in results_df.items():
    plot_anomalies(X_pca, labels, model_name)


# HyperTuning
X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)

# Dictionary to store results
results = {}

# Helper function for plotting anomalies
def plot_anomalies(model, X_test_pca, title, is_clustering=False):
    if is_clustering:
        clusters = model.fit_predict(X_test_pca)
        anomalies = clusters == -1
    else:
        preds = model.fit(X_test).predict(X_test)
        anomalies = preds == -1

    plt.figure(figsize=(8, 6))
    plt.scatter(X_test_pca[~anomalies, 0], X_test_pca[~anomalies, 1], color="blue", s=10, label="Inliers")
    plt.scatter(X_test_pca[anomalies, 0], X_test_pca[anomalies, 1], color="red", s=10, label="Anomalies")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(title)
    plt.legend()
    plt.show()

# Define search spaces
if_params = {
    'n_estimators': [50, 100, 200],
    'max_samples': ['auto', 0.6, 0.8],
    'contamination': [0.1, 0.2, 0.3],
    'max_features': [0.5, 1.0]
}

# Isolation Forest - Grid Search
grid_search_if = GridSearchCV(IsolationForest(random_state=42), if_params, scoring='accuracy', n_jobs=-1)
grid_search_if.fit(X_train)
results["Isolation Forest (Grid Search)"] = grid_search_if.best_estimator_.score_samples(X_test).mean()

# Isolation Forest - Optuna
def objective_if(trial):
    model = IsolationForest(
        n_estimators=trial.suggest_int("n_estimators", 50, 200),
        max_samples=trial.suggest_categorical("max_samples", ["auto", 0.6, 0.8]),
        contamination=trial.suggest_float("contamination", 0.1, 0.3),
        max_features=trial.suggest_float("max_features", 0.5, 1.0),
        random_state=42
    )
    return model.fit(X_train).score_samples(X_test).mean()

print("Starting Optuna for Isolation Forest")
study_if = optuna.create_study(direction="maximize")
study_if.optimize(objective_if, n_trials=10)
results["Isolation Forest (Optuna)"] = study_if.best_value

plt.figure(figsize=(10, 6))
sc = plt.scatter(
    pca_result[:, 0], pca_result[:, 1], 
    c=values, cmap="viridis", edgecolor="k"
)
plt.colorbar(sc, label="Objective Function Value")
plt.title("PCA of Hyperparameters for Isolation Forest")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()

