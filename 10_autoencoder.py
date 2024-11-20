# Autoencoders

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# Load processed data from CSV
processed_data = pd.read_csv('/cleaned_data.csv').values  # .values converts DataFrame to numpy array

# Split data into training and testing sets
X_train, X_test = train_test_split(processed_data, test_size=0.2, random_state=42)

# Build
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)
bottleneck = Dense(16, activation='relu')(encoded)
decoded = Dense(32, activation='relu')(bottleneck)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
output_layer = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train the autoencoder
history = autoencoder.fit(
    X_train, X_train,
    epochs=20, batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Evaluate
reconstructed_test = autoencoder.predict(X_test)
test_errors = np.mean(np.power(X_test - reconstructed_test, 2), axis=1)

# Set threshold for anomaly detection
threshold = np.percentile(test_errors, 95)

# Classify anomalies
y_test_pred = (test_errors > threshold).astype(int)

# Simulated ground truth (adjust as needed)
y_test_actual = np.random.choice([0, 1], size=len(y_test_pred), p=[0.9, 0.1])

# Calculate metrics
accuracy = accuracy_score(y_test_actual, y_test_pred)
precision = precision_score(y_test_actual, y_test_pred)
recall = recall_score(y_test_actual, y_test_pred)
f1 = f1_score(y_test_actual, y_test_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test_actual, y_test_pred)
print("Confusion Matrix:")
print(conf_matrix)



# 1. Plot training and validation loss curves
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 2. Plot reconstruction error distribution
plt.figure(figsize=(10, 6))
sns.histplot(test_errors, kde=True, color='blue', bins=50)
plt.axvline(x=threshold, color='red', linestyle='--', label='Threshold')
plt.title('Reconstruction Error Distribution')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# 3. Plot anomalies in the full dataset (train and test)
# Calculate errors for the entire dataset
reconstructed_data = autoencoder.predict(processed_data)
full_data_errors = np.mean(np.power(processed_data - reconstructed_data, 2), axis=1)

# Classify anomalies
y_full_pred = (full_data_errors > threshold).astype(int)

# Plot anomalies
plt.figure(figsize=(10, 6))
sns.scatterplot(x=np.arange(len(full_data_errors)), y=full_data_errors, hue=y_full_pred, palette={0: 'green', 1: 'red'})
plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
plt.title('Anomalies in Full Dataset')
plt.xlabel('Index')
plt.ylabel('Reconstruction Error')
plt.legend()
plt.show()