import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from keras import layers, models
from keras.optimizers import Adam

# Load and clean the data
data = pd.read_csv('Healthcare_Providers.csv')
data_cleaned = data.drop_duplicates()

# Convert numeric columns to appropriate types
numeric_cols = [
    'Zip Code of the Provider', 'Number of Services', 'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services', 'Average Medicare Allowed Amount',
    'Average Submitted Charge Amount', 'Average Medicare Payment Amount',
    'Average Medicare Standardized Amount'
]
for col in numeric_cols:
    data_cleaned[col] = pd.to_numeric(data_cleaned[col], errors='coerce')

# Encode categorical columns
categorical_cols = [
    'Gender of the Provider', 'Entity Type of the Provider', 'Medicare Participation Indicator',
    'HCPCS Drug Indicator', 'Credentials of the Provider', 'HCPCS Code', 'HCPCS Description'
]
label_encoder = LabelEncoder()
for col in categorical_cols:
    if col in data_cleaned.columns:
        data_cleaned[col] = label_encoder.fit_transform(data_cleaned[col].astype(str))

# One-hot encode additional columns
one_hot_cols = [
    'Provider Type', 'Place of Service', 'State Code of the Provider', 'Country Code of the Provider'
]
data_cleaned = pd.get_dummies(data_cleaned, columns=one_hot_cols, drop_first=True)

# Impute missing values
imputer = SimpleImputer(strategy="median")
data_cleaned[numeric_cols] = imputer.fit_transform(data_cleaned[numeric_cols])

# Normalize and standardize specific columns
normalize_cols = [
    'Number of Services', 'Number of Medicare Beneficiaries', 
    'Number of Distinct Medicare Beneficiary/Per Day Services'
]
standardize_cols = [
    'Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 
    'Average Medicare Payment Amount', 'Average Medicare Standardized Amount'
]

minmax_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

data_cleaned[normalize_cols] = minmax_scaler.fit_transform(data_cleaned[normalize_cols])
data_cleaned[standardize_cols] = standard_scaler.fit_transform(data_cleaned[standardize_cols])

X_train, X_test = train_test_split(data_cleaned[numeric_cols], test_size=0.2, random_state=42)

X_train_scaled = standard_scaler.fit_transform(X_train)
X_test_scaled = standard_scaler.transform(X_test)

input_size = X_train_scaled.shape[1]

autoencoder_model = models.Sequential([
    layers.InputLayer(input_shape=(input_size,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(input_size, activation='sigmoid')
])

autoencoder_model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')

training_history = autoencoder_model.fit(
    X_train_scaled, X_train_scaled,
    epochs=100,
    batch_size=64,
    shuffle=True,
    validation_data=(X_test_scaled, X_test_scaled),
    validation_split=0.1
)

# Reconstruction
X_test_reconstructed = autoencoder_model.predict(X_test_scaled)
reconstruction_error = np.mean(np.abs(X_test_scaled - X_test_reconstructed), axis=1)

# Threshold
error_threshold = np.percentile(reconstruction_error, 95)
autoencoder_anomalies = reconstruction_error > error_threshold

# Isolation Forest
isolation_model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
isolation_labels = isolation_model.fit_predict(reconstruction_error.reshape(-1, 1))
isolation_anomalies = (isolation_labels == -1)

#check anomaly for a specific row
def check_anomaly(row_index):
    row = X_test_scaled[row_index]
    reconstructed_row = autoencoder_model.predict(row.reshape(1, -1))
    error = np.mean(np.abs(row - reconstructed_row))
    is_anomaly = error > error_threshold
    print(f"Row Index: {row_index}")
    print(f"Reconstruction Error: {error:.4f}")
    print(f"Threshold: {error_threshold:.4f}")
    print(f"Anomaly: {'Yes' if is_anomaly else 'No'}")
    return is_anomaly

accuracy = accuracy_score(isolation_anomalies, autoencoder_anomalies)
precision = precision_score(isolation_anomalies, autoencoder_anomalies)
recall = recall_score(isolation_anomalies, autoencoder_anomalies)
f1 = f1_score(isolation_anomalies, autoencoder_anomalies)
total_anomalies=np.sum(autoencoder_anomalies)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Total Anomalies:{total_anomalies}")

# Confusion Matrix
conf_matrix = confusion_matrix(isolation_anomalies, autoencoder_anomalies)
print(f"Confusion Matrix:\n{conf_matrix}")

# Visualizations
plt.figure(figsize=(12, 6))

# Training and Validation Loss
plt.subplot(1, 2, 1)
plt.plot(training_history.history['loss'], label='Training Loss', color='blue')
plt.plot(training_history.history['val_loss'], label='Validation Loss', color='green')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Autoencoder Training vs Validation Loss")
plt.legend()

# Reconstruction Error Distribution
plt.subplot(1, 2, 2)
plt.hist(reconstruction_error, bins=30, alpha=0.7, color='purple', label='Reconstruction Error')
plt.axvline(error_threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.title("Reconstruction Error Distribution")
plt.legend()

plt.tight_layout()
plt.show()

# Input row number to check anomaly
row_number = int(input("Enter row number to check for anomaly: "))
check_anomaly(row_number)
