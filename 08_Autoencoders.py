# Autoencoders

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
# %matplotlib inline


# Load the data
data = pd.read_csv("encoded.csv")

# Select features for anomaly detection
features = [
    'Number of Services', 'Number of Medicare Beneficiaries', 
    'Number of Distinct Medicare Beneficiary/Per Day Services', 
    'Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 
    'Average Medicare Payment Amount', 'Average Medicare Standardized Amount'
]

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features])

# Split data into training (normal) and testing (mixed) sets
# Assuming that the majority of data is normal, we'll use all data for training
train_data = data_scaled

# Define the autoencoder model
input_dim = train_data.shape[1]
encoding_dim = 4  # Smaller dimensions to capture essential features

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="relu")(input_layer)
decoder = Dense(input_dim, activation="linear")(encoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)

# Compile the model
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

# Train the autoencoder on the training data
history = autoencoder.fit(
    train_data, train_data,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    shuffle=True
)

# Calculate reconstruction error on all data
reconstructed_data = autoencoder.predict(data_scaled)
reconstruction_error = np.mean(np.square(data_scaled - reconstructed_data), axis=1)

# Set a threshold for anomaly detection
# Here, we use the 95th percentile of the reconstruction error as the threshold
threshold = np.percentile(reconstruction_error, 95)
data["anomaly"] = reconstruction_error > threshold

# Separate normal and anomaly data
normal_data = data[data["anomaly"] == False]
anomalies = data[data["anomaly"] == True]

# Print total number of anomalies detected
total_anomalies = len(anomalies)
print(f"Total number of anomalies detected: {total_anomalies}")

# Plot the reconstruction error

plt.figure(figsize=(10, 6))
plt.hist(reconstruction_error, bins=50, color="blue", label="Reconstruction Error")
plt.axvline(threshold, color="red", linestyle="--", label="Threshold")
plt.title("Reconstruction Error Distribution")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Visualize anomalies vs normal data on selected features
# Define axis limits based on percentiles
x_min, x_max = normal_data['Number of Services'].quantile(0.01), normal_data['Number of Services'].quantile(0.99)
y_min, y_max = normal_data['Average Medicare Payment Amount'].quantile(0.01), normal_data['Average Medicare Payment Amount'].quantile(0.99)

plt.figure(figsize=(10, 6))

# Plot normal data in blue
plt.scatter(normal_data['Number of Services'], normal_data['Average Medicare Payment Amount'], 
            c='blue', label='Normal', alpha=0.6)
# Plot anomalies in red
plt.scatter(anomalies['Number of Services'], anomalies['Average Medicare Payment Amount'], 
            c='red', label='Anomaly', alpha=0.6)

plt.title('Anomaly Detection using Autoencoder')
plt.xlabel('Number of Services')
plt.ylabel('Average Medicare Payment Amount')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend()
plt.show()