import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Load the dataset
data_path = 'path/to/your/file.csv'  # Update this to your file path
df = pd.read_csv(data_path)

# Normalize the data
scaler = StandardScaler()
data_normalized = scaler.fit_transform(df)

# Split the data into training and testing sets
X_train, X_test = train_test_split(data_normalized, test_size=0.2, random_state=42)

# Define the Autoencoder model
input_dim = X_train.shape[1]
encoding_dim = 32  # Dimension of the compressed representation

# Encoder
input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(encoding_dim, activation='relu')(encoded)

# Decoder
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(input_dim, activation='linear')(decoded)

# Autoencoder model
autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
history = autoencoder.fit(X_train, X_train,
                          epochs=50,
                          batch_size=256,
                          shuffle=True,
                          validation_split=0.2,
                          verbose=1)

# Reconstruction error threshold calculation
X_train_pred = autoencoder.predict(X_train)
train_reconstruction_error = np.mean(np.square(X_train - X_train_pred), axis=1)
threshold = np.percentile(train_reconstruction_error, 95)  # 95th percentile as the threshold for anomalies

# Test data predictions and anomaly detection
X_test_pred = autoencoder.predict(X_test)
test_reconstruction_error = np.mean(np.square(X_test - X_test_pred), axis=1)
y_test_pred = (test_reconstruction_error > threshold).astype(int)  # 1 for anomaly, 0 for normal

### OPTION 1: Manual Labeling for Evaluation Metrics ###
# Uncomment and modify the following if you know which samples are anomalies
# Example: First 100 entries in the test set are anomalies
# y_test = np.zeros(X_test.shape[0])
# y_test[:100] = 1  # Label the first 100 samples as anomalies

# If y_test is defined, calculate evaluation metrics
# if 'y_test' in locals():
#     accuracy = accuracy_score(y_test, y_test_pred)
#     precision = precision_score(y_test, y_test_pred)
#     recall = recall_score(y_test, y_test_pred)
#     f1 = f1_score(y_test, y_test_pred)
#     conf_matrix = confusion_matrix(y_test, y_test_pred)
#     print("Accuracy:", accuracy)
#     print("Precision:", precision)
#     print("Recall:", recall)
#     print("F1 Score:", f1)
#     print("Confusion Matrix:\n", conf_matrix)

### OPTION 2: Unsupervised Evaluation Without y_test ###
# Analyze reconstruction errors without true labels
print("Mean Reconstruction Error:", np.mean(test_reconstruction_error))
print("Max Reconstruction Error:", np.max(test_reconstruction_error))
print("Anomalies Detected:", np.sum(y_test_pred), "out of", len(y_test_pred))