import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
from sklearn.metrics import confusion_matrix


# Define numeric columns
numeric_columns = [
    'Number of Services', 'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services', 'Average Medicare Allowed Amount',
    'Average Submitted Charge Amount', 'Average Medicare Payment Amount',
    'Average Medicare Standardized Amount'
]

# Normalize and standardize selected columns
columns_to_normalize = [
    'Number of Services', 'Number of Medicare Beneficiaries', 'Number of Distinct Medicare Beneficiary/Per Day Services'
]
columns_to_standardize = [
    'Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 
    'Average Medicare Payment Amount', 'Average Medicare Standardized Amount'
]

# MinMax scaling and Standard scaling
minmax_scaler = MinMaxScaler()
normalized_df = pd.DataFrame(minmax_scaler.fit_transform(df[columns_to_normalize]), columns=columns_to_normalize)

standard_scaler = StandardScaler()
standardized_df = pd.DataFrame(standard_scaler.fit_transform(df[columns_to_standardize]), columns=columns_to_standardize)

# Combine into final dataset
final_dataset = pd.concat([df.drop(columns=columns_to_normalize + columns_to_standardize),
                           normalized_df, standardized_df], axis=1)

# Split data into training and testing sets
X_train, X_test = train_test_split(final_dataset[numeric_columns], test_size=0.2, random_state=42)

# Define Autoencoder model
input_dim = X_train.shape[1]
encoding_dim = int(input_dim * 1.5)  # Increased capacity

autoencoder = Sequential([
    # Encoder
    Dense(encoding_dim, activation="relu", input_shape=(input_dim,)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(int(encoding_dim/2), activation="relu"),
    BatchNormalization(),
    Dropout(0.2),
    Dense(int(encoding_dim/4), activation="relu"),
    BatchNormalization(),
    
    # Decoder
    Dense(int(encoding_dim/2), activation="relu"),
    BatchNormalization(),
    Dense(encoding_dim, activation="relu"),
    BatchNormalization(),
    Dense(input_dim, activation="linear")
])

autoencoder.compile(optimizer="adam", loss="mse")

# Add Early Stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train the Autoencoder
history = autoencoder.fit(
    X_train, X_train,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Calculate reconstruction error on the test set
X_test_pred = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - X_test_pred, 2), axis=1)

# Set a threshold for anomaly detection
threshold = mse.mean() + mse.std()
anomaly_labels = (mse > threshold).astype(int)

# Add anomalies to the dataset
final_dataset['anomaly_autoencoder'] = 0
final_dataset.loc[X_test.index, 'anomaly_autoencoder'] = anomaly_labels

# Separate normal data and anomalies
anomalies_autoencoder = final_dataset[final_dataset['anomaly_autoencoder'] == 1]
normal_data_autoencoder = final_dataset[final_dataset['anomaly_autoencoder'] == 0]

# Display anomalies
print("Anomalies detected by Autoencoder:")
print(anomalies_autoencoder.head())
print(f"Total number of anomalies detected by Autoencoder: {anomalies_autoencoder.shape[0]}")

# Plotting anomalies vs normal data
plt.figure(figsize=(10, 6))
plt.scatter(normal_data_autoencoder['Number of Services'], normal_data_autoencoder['Average Medicare Payment Amount'], 
            color='blue', label='Normal', alpha=0.5)
plt.scatter(anomalies_autoencoder['Number of Services'], anomalies_autoencoder['Average Medicare Payment Amount'], 
            color='purple', label='Anomalies (Autoencoder)', alpha=0.7)
plt.xlabel('Number of Services')
plt.ylabel('Average Medicare Payment Amount')
plt.title('Anomaly Detection with Autoencoder')
plt.legend()
plt.grid()
plt.show()

# Plot loss during training
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save the final dataset with Autoencoder anomaly labels
final_dataset.to_csv('autoencoder_anomalies.csv', index=False)
