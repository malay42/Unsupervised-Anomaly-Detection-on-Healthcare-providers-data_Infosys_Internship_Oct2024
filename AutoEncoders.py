import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

df = pd.read_csv('Healthcare_Providers.csv')        # Loading dataset

# Load dataset
data = df.values

# Split data into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Step 1: Train Isolation Forest for anomaly detection
isolation_forest = IsolationForest(n_estimators = 165, max_samples = 0.6, contamination = 0.05,
                            max_features = 0.9368446042993135, random_state=42)  # Assuming 5% contamination
isolation_forest.fit(train_data)
iso_forest_pred = isolation_forest.predict(test_data)

# Convert Isolation Forest predictions to binary format (1 for normal, 0 for anomaly)
iso_forest_labels = np.where(iso_forest_pred == 1, 0, 1)  # 1: anomaly, 0: normal

# Step 2: Define and Train Autoencoder
input_dim = train_data.shape[1]  # Number of features

# Autoencoder architecture
input_layer = Input(shape=(input_dim,))
encoded = Dense(10, activation='relu')(input_layer)
encoded = Dense(5, activation='relu')(encoded)
encoded = Dense(2, activation='relu')(encoded)  # Bottleneck layer

decoded = Dense(5, activation='relu')(encoded)
decoded = Dense(10, activation='relu')(decoded)
decoded = Dense(input_dim, activation='linear')(decoded)

# Compile autoencoder
autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the autoencoder
autoencoder.fit(
    train_data, train_data,
    epochs=50,
    batch_size=64,
    shuffle=True,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# Step 3: Use the Autoencoder to detect anomalies based on reconstruction error
reconstructed_test = autoencoder.predict(test_data)
reconstruction_error_test = np.mean(np.power(test_data - reconstructed_test, 2), axis=1)

# Determine threshold based on training reconstruction error (95th percentile)
threshold = np.percentile(reconstruction_error_test, 95)

# Predict anomalies based on autoencoder reconstruction error
autoencoder_labels = np.where(reconstruction_error_test > threshold, 1, 0)  # 1: anomaly, 0: normal

# Step 4: Confusion Matrix and Performance Metrics
# Compare Isolation Forest labels (as ground truth) with Autoencoder results
cm = confusion_matrix(iso_forest_labels, autoencoder_labels)
print("Confusion Matrix:")
print(cm)

# Detailed classification report
print("Classification Report:")
print(classification_report(iso_forest_labels, autoencoder_labels, target_names=["Normal", "Anomaly"]))
