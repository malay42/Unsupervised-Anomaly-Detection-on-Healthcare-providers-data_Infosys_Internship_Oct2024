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
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

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

# Prepare the dataset and split into train/test
X = df[numeric_columns].values
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Define the final Autoencoder architecture
input_dim = X.shape[1]
encoder_input = Input(shape=(input_dim,))
encoder = Dense(256, activation='relu')(encoder_input)
encoder = Dropout(0.0)(encoder)
encoder = Dense(128, activation='relu')(encoder)
encoder = Dropout(0.0)(encoder)
encoder = Dense(64, activation='relu')(encoder)
encoder = Dropout(0.0)(encoder)
encoder = Dense(32, activation='relu')(encoder)
encoder = BatchNormalization()(encoder)
encoder_output = Dense(32, activation='relu')(encoder)

# Define Decoder
decoder = Dense(32, activation='relu')(encoder_output)
decoder = Dropout(0.0)(decoder)
decoder = Dense(64, activation='relu')(decoder)
decoder = Dropout(0.0)(decoder)
decoder = Dense(128, activation='relu')(decoder)
decoder = Dropout(0.0)(decoder)
decoder = Dense(256, activation='relu')(decoder)
decoder = Dropout(0.0)(decoder)
decoder_output = Dense(input_dim, activation='linear')(decoder)

# Combine into Autoencoder
autoencoder = Model(inputs=encoder_input, outputs=decoder_output)
autoencoder.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')

# Print model summaries
encoder_model = Model(inputs=encoder_input, outputs=encoder_output)
print("Encoder Summary:")
encoder_model.summary()

decoder_model = Model(inputs=encoder_output, outputs=decoder_output)
print("\nDecoder Summary:")
decoder_model.summary()

# Train the autoencoder with validation split
history = autoencoder.fit(
    X_train, X_train,
    epochs=20,
    batch_size=32,
    shuffle=True,
    validation_data=(X_test, X_test),
    verbose=1
)

# Plot training and validation loss
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid()
plt.show()

# Predict and calculate reconstruction error
X_pred = autoencoder.predict(X)
reconstruction_error = np.mean(np.square(X - X_pred), axis=1)

# Isolation Forest and One-Class SVM anomaly detection
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_labels = iso_forest.fit_predict(reconstruction_error.reshape(-1, 1))

one_class_svm = OneClassSVM(gamma='auto', nu=0.05)
svm_labels = one_class_svm.fit_predict(reconstruction_error.reshape(-1, 1))

# Find best threshold
best_threshold = None
best_f1 = 0
thresholds = np.arange(94, 96.5, 0.5)
scores = []

for percentile in thresholds:
    anomaly_threshold = np.percentile(reconstruction_error, percentile)
    predicted_anomalies = (reconstruction_error > anomaly_threshold).astype(int)
    
    accuracy = accuracy_score(iso_labels == -1, predicted_anomalies)
    precision = precision_score(iso_labels == -1, predicted_anomalies)
    recall = recall_score(iso_labels == -1, predicted_anomalies)
    f1 = f1_score(iso_labels == -1, predicted_anomalies)
    
    scores.append({'Threshold Percentile': percentile, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1})
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = anomaly_threshold

# Convert scores to DataFrame and plot metrics
scores_df = pd.DataFrame(scores)

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

# Final anomaly detection
predicted_anomalies = (reconstruction_error > best_threshold).astype(int)
anomalies = df[predicted_anomalies == 1]
print("Detected Anomalies:")
print(anomalies)

# Plot confusion matrix
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

# Function to detect anomalies in new data
def detect_anomaly_in_row(row, scaler, autoencoder, best_threshold):
    row_data = row[numeric_columns].values.reshape(1, -1)
    row_data = scaler.transform(row_data)
    row_pred = autoencoder.predict(row_data)
    row_error = np.mean(np.square(row_data - row_pred))
    is_anomaly = row_error > best_threshold
    return is_anomaly, row_error
'''
# Test the anomaly detection function
file_path = "C:/Users/shiva/Desktop/project/anomalies_report.csv"
df = pd.read_csv(file_path)
random_row_index = np.random.randint(0, len(df))
random_row = df.iloc[random_row_index]
is_anomaly, row_error = detect_anomaly_in_row(random_row, scaler, autoencoder, best_threshold)

print(f"Row index {random_row_index} - Reconstruction Error: {row_error:.4f}")
print("Anomaly" if is_anomaly else "Normal")
'''
# Function to detect anomalies for a specific row number entered by the user
def detect_anomaly_for_row(df, scaler, autoencoder, best_threshold, numeric_columns, row_number):
    # Validate the row number
    if row_number < 0 or row_number >= len(df):
        raise ValueError(f"Row number {row_number} is out of range. Please enter a number between 0 and {len(df)-1}.")
    
    # Select the specific row
    selected_row = df.iloc[row_number]
    
    # Scale the numeric features of the row
    row_data = selected_row[numeric_columns].values.reshape(1, -1)
    row_data_scaled = scaler.transform(row_data)
    
    # Predict with autoencoder and calculate reconstruction error
    row_pred = autoencoder.predict(row_data_scaled)
    row_error = np.mean(np.square(row_data_scaled - row_pred))
    
    # Determine if the row is an anomaly
    is_anomaly = row_error > best_threshold
    
    return selected_row, is_anomaly, row_error

# Prompt the user to enter a row number
try:
    row_number = int(input(f"Enter a row number (0 to {len(df)-1}) for anomaly detection: "))
    selected_row, is_anomaly, row_error = detect_anomaly_for_row(df, scaler, autoencoder, best_threshold, numeric_columns, row_number)
    
    # Print results
    print(f"\nSelected Row Number: {row_number}")
    print(f"Selected Row Data:\n{selected_row}")
    print(f"Reconstruction Error: {row_error:.4f}")
    print("Anomaly Detected!" if is_anomaly else "Normal")
except ValueError as e:
    print(f"Error: {e}")

# Visualize PCA results with anomaly labels
X_test_pred = autoencoder.predict(X_test)
reconstruction_error_test = np.mean(np.square(X_test - X_test_pred), axis=1)
test_predicted_anomalies = (reconstruction_error_test > best_threshold).astype(int)

pca_test_features = pca.fit_transform(X_test)
df_pca_test = pd.DataFrame(pca_test_features, columns=['PCA1', 'PCA2'])
df_pca_test['Anomaly'] = test_predicted_anomalies

clean_data = df_pca_test[df_pca_test['Anomaly'] == 0]
fraud_data = df_pca_test[df_pca_test['Anomaly'] == 1]

plt.figure(figsize=(8, 6))
plt.scatter(clean_data['PCA1'], clean_data['PCA2'], c='blue', label='Clean Data', alpha=0.5)
plt.scatter(fraud_data['PCA1'], fraud_data['PCA2'], c='red', label='Fraud Data', alpha=0.7)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('Clean Data vs Fraud Data (Testing Set)')
plt.legend()
plt.grid()
plt.show()

# Plot histogram of reconstruction errors
clean_reconstruction_errors = reconstruction_error_test[test_predicted_anomalies == 0]
fraud_reconstruction_errors = reconstruction_error_test[test_predicted_anomalies == 1]

plt.figure(figsize=(8, 6))
plt.hist(clean_reconstruction_errors, bins=30, color='blue', alpha=0.5, label='Clean Data')
plt.hist(fraud_reconstruction_errors, bins=30, color='red', alpha=0.7, label='Fraud Data')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.title('Histogram of Reconstruction Errors (Clean vs Fraud)')
plt.legend()
plt.grid()
plt.show()