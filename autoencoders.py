import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam

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

# Polynomial feature transformation
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
poly_features = poly.fit_transform(df[numeric_columns])
poly_feature_names = poly.get_feature_names_out(numeric_columns)
df_poly = pd.DataFrame(poly_features, columns=poly_feature_names)

# PCA transformation for visualization
pca = PCA(n_components=2)
pca_features = pca.fit_transform(df_poly)
df_pca = pd.DataFrame(pca_features, columns=['PCA1', 'PCA2'])

# Prepare the dataset for the autoencoder
X = df[numeric_columns].values

# Define the Autoencoder Model with additional layers and lower dropout
input_dim = X.shape[1]
encoding_dim = 6  # Increased encoding dimension for better feature capture

input_layer = Input(shape=(input_dim,))
encoder = Dense(12, activation="relu")(input_layer)  # Additional layer
encoder = Dense(encoding_dim, activation="relu")(encoder)
encoder = Dropout(0.1)(encoder)  # Lower dropout rate
decoder = Dense(12, activation="relu")(encoder)  # Symmetrical layer
decoder = Dense(input_dim, activation="linear")(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')  # Reduced learning rate

# Train the autoencoder with increased epochs
history = autoencoder.fit(X, X, epochs=20, batch_size=32, shuffle=True, validation_split=0.2, verbose=1)

# Predict and calculate reconstruction error
X_pred = autoencoder.predict(X)
reconstruction_error = np.mean(np.square(X - X_pred), axis=1)

# Isolation Forest and One-Class SVM anomaly detection
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_labels = iso_forest.fit_predict(reconstruction_error.reshape(-1, 1))

one_class_svm = OneClassSVM(gamma='auto', nu=0.05)
svm_labels = one_class_svm.fit_predict(reconstruction_error.reshape(-1, 1))

# Using the best threshold with a more granular search to maximize F1 Score
best_threshold = None
best_f1 = 0
thresholds = np.arange(94, 96.5, 0.5)  # Granular search from 94% to 96% in 0.5% increments
scores = []

for percentile in thresholds:
    anomaly_threshold = np.percentile(reconstruction_error, percentile)
    predicted_anomalies = (reconstruction_error > anomaly_threshold).astype(int)

    # Calculate accuracy metrics for the current threshold using Isolation Forest
    accuracy = accuracy_score(iso_labels == -1, predicted_anomalies)
    precision = precision_score(iso_labels == -1, predicted_anomalies)
    recall = recall_score(iso_labels == -1, predicted_anomalies)
    f1 = f1_score(iso_labels == -1, predicted_anomalies)
    
    scores.append({'Threshold Percentile': percentile, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1})
    
    # Track the best threshold based on F1 score
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = anomaly_threshold

# Convert scores to DataFrame for easy visualization
scores_df = pd.DataFrame(scores)

# Plot Accuracy, Precision, Recall, and F1 Score for each threshold
plt.figure(figsize=(10, 6))
plt.plot(scores_df['Threshold Percentile'], scores_df['Accuracy'], label="Accuracy", marker='o')
plt.plot(scores_df['Threshold Percentile'], scores_df['Precision'], label="Precision", marker='o')
plt.plot(scores_df['Threshold Percentile'], scores_df['Recall'], label="Recall", marker='o')
plt.plot(scores_df['Threshold Percentile'], scores_df['F1 Score'], label="F1 Score", marker='o')
plt.xlabel("Threshold Percentile")
plt.ylabel("Score")
plt.title("Accuracy, Precision, Recall, and F1 Score vs. Threshold Percentile")
plt.legend()
plt.grid()
plt.show()

print(f"Best Threshold: {best_threshold:.2f} with F1 Score: {best_f1:.2f}")

# Re-run anomaly detection with the best threshold
predicted_anomalies = (reconstruction_error > best_threshold).astype(int)

# Extract anomalies for review
anomalies = df[predicted_anomalies == 1]
print("Detected Anomalies:")
print(anomalies)

# Confusion Matrix with the best threshold
cm = confusion_matrix(iso_labels == -1, predicted_anomalies)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix at Best Threshold")
plt.show()

# Save anomalies to CSV
def save_anomalies_to_csv(anomalies_df, csv_path="anomalies_report.csv"):
    anomalies_df.to_csv(csv_path, index=False)
    print(f"Anomalies saved to {csv_path}")

save_anomalies_to_csv(anomalies)

import pandas as pd
import numpy as np

# Function to detect if a given row is an anomaly
def detect_anomaly_in_row(row, scaler, autoencoder, best_threshold):
    # Select numeric columns for the row, matching the model's input format
    row_data = row[numeric_columns].values.reshape(1, -1)
    
    # Standardize the row data
    row_data = scaler.transform(row_data)
    
    # Predict with the autoencoder
    row_pred = autoencoder.predict(row_data)
    
    # Calculate reconstruction error
    row_error = np.mean(np.square(row_data - row_pred))
    
    # Determine if the row is an anomaly based on the threshold
    is_anomaly = row_error > best_threshold
    return is_anomaly, row_error

# Example usage of the function
file_path = "C:/Users/shiva/Desktop/project/anomalies_report.csv"
df = pd.read_csv(file_path)

# Choose a random row index
random_row_index = np.random.randint(0, len(df))
random_row = df.iloc[random_row_index]

# Detect if the random row is an anomaly
is_anomaly, row_error = detect_anomaly_in_row(random_row, scaler, autoencoder, best_threshold)

print(f"Row index {random_row_index} - Reconstruction Error: {row_error:.4f}")
print("Anomaly" if is_anomaly else "Normal")
 