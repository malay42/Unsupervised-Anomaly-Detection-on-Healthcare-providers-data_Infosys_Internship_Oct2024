import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
import category_encoders as ce
import optuna

# Load the dataset
file_path = "C:/Users/shiva/Desktop/project/Cleaned_Healthcare_Providers.csv"
df = pd.read_csv(file_path)

# Select numeric columns for analysis
numeric_columns = [
    'Number of Services', 'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services',
    'Average Medicare Allowed Amount', 'Average Submitted Charge Amount',
    'Average Medicare Payment Amount', 'Average Medicare Standardized Amount'
]

# Standardize numeric columns
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Apply Binary Encoding for categorical features
binary_encoding_columns = ['Provider Type', 'Credentials of the Provider', 'Gender of the Provider', 'Entity Type of the Provider']
binary_encoder = ce.BinaryEncoder(cols=binary_encoding_columns)
df_encoded = binary_encoder.fit_transform(df)

# Further encode any remaining categorical columns
label_encoding_columns = ['Medicare Participation Indicator', 'Place of Service', 'HCPCS Code', 'HCPCS Drug Indicator']
for col in label_encoding_columns:
    df_encoded[col] = df_encoded[col].astype('category').cat.codes

# Drop any non-numeric columns and NaN values
df_encoded = df_encoded.apply(pd.to_numeric, errors='coerce').dropna(axis=1)

# Standardize the cleaned dataset
scaled_df = scaler.fit_transform(df_encoded)

# Sample the data for faster processing
sample_size = 5000  # Adjust based on your dataset size
df_sampled = df_encoded.sample(n=sample_size, random_state=42)
scaled_sampled_df = scaler.fit_transform(df_sampled)

# Apply PCA to the sampled data
pca = PCA(n_components=2, random_state=42)
pca_components = pca.fit_transform(scaled_sampled_df)
pca_df = pd.DataFrame(pca_components, columns=['PCA1', 'PCA2'])

# ---------- Hyperparameter Tuning with Optuna for Isolation Forest ----------
def objective(trial):
    # Hyperparameters for Isolation Forest
    n_estimators = trial.suggest_int('n_estimators', 100, 300)
    max_samples = trial.suggest_float('max_samples', 0.1, 1.0)
    contamination = trial.suggest_float('contamination', 0.01, 0.1)

    model = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, contamination=contamination, random_state=42)
    model.fit(scaled_sampled_df)
    
    # Evaluate the model with the number of anomalies detected
    labels = model.predict(scaled_sampled_df)
    n_anomalies = np.sum(labels == -1)
    
    return n_anomalies  # Minimize number of detected anomalies

# Create an Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)  # Limit to 20 trials

# Retrieve best parameters
best_params_iso = study.best_params
print("Best parameters for Isolation Forest:", best_params_iso)

# Train the final model with the best parameters
iso_forest = IsolationForest(**best_params_iso, random_state=42)
iso_labels = iso_forest.fit_predict(scaled_sampled_df)
pca_df['Isolation_Forest'] = np.where(iso_labels == -1, 'Anomaly', 'Normal')

# One-Class SVM Training
svm_model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)  # You can adjust nu if needed
svm_labels = svm_model.fit_predict(scaled_sampled_df)
pca_df['OneClassSVM'] = np.where(svm_labels == -1, 'Anomaly', 'Normal')

# Local Outlier Factor Training
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)  # Adjust parameters as needed
lof_labels = lof.fit_predict(scaled_sampled_df)
pca_df['LocalOutlierFactor'] = np.where(lof_labels == -1, 'Anomaly', 'Normal')

# ---------- Visualization ----------
def plot_results(df, x_col, y_col, label_col, title):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x_col, y=y_col, hue=label_col, style=label_col,
                    markers={'Normal': 'o', 'Anomaly': 'X'},
                    palette={'Normal': 'blue', 'Anomaly': 'red'}, data=df)
    plt.title(title)
    plt.legend(title=label_col, loc='upper right')
    plt.show()

# Plot the results
plot_results(pca_df, 'PCA1', 'PCA2', 'Isolation_Forest', "Isolation Forest (PCA)")
plot_results(pca_df, 'PCA1', 'PCA2', 'OneClassSVM', "One-Class SVM (PCA)")
plot_results(pca_df, 'PCA1', 'PCA2', 'LocalOutlierFactor', "Local Outlier Factor (PCA)")
