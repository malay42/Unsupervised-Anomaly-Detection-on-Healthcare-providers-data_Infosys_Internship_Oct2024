import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import numpy as np
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
file_path = "C:/Users/shiva/Desktop/project/Cleaned_Healthcare_Providers.csv"
df = pd.read_csv(file_path)

# Sample the dataset
df_sampled = df.sample(n=1000, random_state=42)  # Adjust n as needed

# Print sampled DataFrame columns for verification
print("Sampled DataFrame columns:", df_sampled.columns)

# One-Hot Encoding
categorical_cols = [
    'Gender of the Provider',
    'Entity Type of the Provider',
    'Provider Type',
    'Medicare Participation Indicator',
    'Place of Service',
    'HCPCS Code',
    'HCPCS Description',
    'HCPCS Drug Indicator'
]

# Initialize the OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, drop='first')  # Updated for new versions

# Fit and transform the encoder
encoded_df = encoder.fit_transform(df_sampled[categorical_cols])

# Create a DataFrame with the encoded features
encoded_df = pd.DataFrame(encoded_df, columns=encoder.get_feature_names_out(categorical_cols))

# Concatenate the encoded DataFrame with the original DataFrame
encoded_df = pd.concat([df_sampled.reset_index(drop=True), encoded_df], axis=1)

# Drop original categorical columns
encoded_df.drop(columns=categorical_cols, inplace=True)

# Print Encoded DataFrame columns for verification
print("Encoded DataFrame columns:", encoded_df.columns)

# Scaling the numeric features
scaler = StandardScaler()
# Identify numeric columns for scaling
numeric_cols = encoded_df.select_dtypes(include=[np.number]).columns.tolist()
print("Numeric columns for scaling:", numeric_cols)

# Fit and transform the scaler
scaled_sampled_df = scaler.fit_transform(encoded_df[numeric_cols])
scaled_sampled_df = pd.DataFrame(scaled_sampled_df, columns=numeric_cols)

# Perform PCA
pca = PCA(n_components=0.95)  # Adjust n_components as needed
pca_result = pca.fit_transform(scaled_sampled_df)

# Visualize the PCA results
plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
plt.title('PCA of Sampled Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()
plt.show()

# Define the objective function for Hyperopt
def hyperopt_objective(params):
    contamination = params['contamination']
    max_samples = params['max_samples']
    n_estimators = int(params['n_estimators'])  # Convert to int here

    model = IsolationForest(contamination=contamination, max_samples=max_samples, n_estimators=n_estimators)
    model.fit(pca_result)
    
    # Get anomaly scores
    scores = model.score_samples(pca_result)

    return {'loss': -np.mean(scores), 'status': STATUS_OK}

# Define the search space for Hyperopt
space = {
    'contamination': hp.uniform('contamination', 0.01, 0.1),
    'max_samples': hp.uniform('max_samples', 0.1, 1.0),
    'n_estimators': hp.quniform('n_estimators', 100, 500, 10)  # This returns float
}

# Run Hyperopt
trials = Trials()
best = fmin(fn=hyperopt_objective, space=space, algo=tpe.suggest, max_evals=20, trials=trials)

# Print the best parameters found
print("Best parameters found:", best)

# Train the IsolationForest with the best parameters
model = IsolationForest(
    contamination=best['contamination'],
    max_samples=best['max_samples'],
    n_estimators=int(best['n_estimators'])
)

model.fit(pca_result)

# Predict anomalies
anomalies = model.predict(pca_result)
anomalies = np.where(anomalies == -1, 1, 0)  # Convert -1 to 1 (anomaly), 1 to 0 (normal)

# Visualize the anomalies on PCA plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=anomalies, palette=['blue', 'red'], alpha=0.5)
plt.title('PCA with Anomalies Highlighted')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Anomaly', labels=['Normal', 'Anomaly'])
plt.grid()
plt.show()
