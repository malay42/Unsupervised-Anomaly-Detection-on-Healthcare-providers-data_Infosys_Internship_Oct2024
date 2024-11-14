# Training Autoencoders

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

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

# Split data into training (80%) and testing (20%) sets
train_data, test_data = train_test_split(data_scaled, test_size=0.2, random_state=42)

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
    epochs=20,  # Reduced epochs for faster training
    batch_size=32,
    validation_split=0.1,
    shuffle=True
)

# Calculate reconstruction error on the training data
reconstructed_train_data = autoencoder.predict(train_data)
reconstruction_error_train = np.mean(np.square(train_data - reconstructed_train_data), axis=1)

# Set a threshold for anomaly detection (95th percentile on training set)
threshold_train = np.percentile(reconstruction_error_train, 95)

# Identify anomalies in the training data
train_anomalies = reconstruction_error_train > threshold_train

# Generate placeholder true labels for demonstration
# Assuming 5% of the training set are true anomalies
true_labels_train = np.zeros(len(train_data), dtype=int)
anomaly_indices_train = np.random.choice(len(train_data), int(0.05 * len(train_data)), replace=False)
true_labels_train[anomaly_indices_train] = 1

# Calculate evaluation metrics
accuracy_train = accuracy_score(true_labels_train, train_anomalies)
precision_train = precision_score(true_labels_train, train_anomalies)
recall_train = recall_score(true_labels_train, train_anomalies)
f1_train = f1_score(true_labels_train, train_anomalies)

# Print metrics for training set
print(f"Training Set Accuracy: {accuracy_train:.2f}")
print(f"Training Set Precision: {precision_train:.2f}")
print(f"Training Set Recall: {recall_train:.2f}")
print(f"Training Set F1 Score: {f1_train:.2f}")

# Plot the reconstruction error for training set
plt.figure(figsize=(10, 6))
plt.hist(reconstruction_error_train, bins=50, color="blue", label="Reconstruction Error")
plt.axvline(threshold_train, color="red", linestyle="--", label="Threshold")
plt.title("Reconstruction Error Distribution (Training Set)")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Visualize anomalies vs normal data on selected features in training set
train_data_df = pd.DataFrame(train_data, columns=features)
train_data_df["anomaly"] = train_anomalies

# Separate normal and anomaly data in training set for visualization
normal_train_data = train_data_df[train_data_df["anomaly"] == False]
anomaly_train_data = train_data_df[train_data_df["anomaly"] == True]

# Define axis limits based on percentiles
x_min, x_max = normal_train_data['Number of Services'].quantile(0.01), normal_train_data['Number of Services'].quantile(0.99)
y_min, y_max = normal_train_data['Average Medicare Payment Amount'].quantile(0.01), normal_train_data['Average Medicare Payment Amount'].quantile(0.99)

plt.figure(figsize=(10, 6))

# Plot normal data in blue
plt.scatter(normal_train_data['Number of Services'], normal_train_data['Average Medicare Payment Amount'], 
            c='blue', label='Normal', alpha=0.6)
# Plot anomalies in red
plt.scatter(anomaly_train_data['Number of Services'], anomaly_train_data['Average Medicare Payment Amount'], 
            c='red', label='Anomaly', alpha=0.6)

plt.title('Anomaly Detection using Autoencoder (Training Set)')
plt.xlabel('Number of Services')
plt.ylabel('Average Medicare Payment Amount')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend()
plt.show()

# Count the number of anomalies in the training set
num_anomalies_train = np.sum(train_anomalies)
print(f"Number of anomalies detected in the training set: {num_anomalies_train}")


# Get the row numbers of the anomalies in the training set
anomaly_row_numbers = np.where(train_anomalies)[0]  # Row numbers where train_anomalies is True
num_anomalies = len(anomaly_row_numbers)  # Count of anomalies

print(f"Row numbers of anomalies in the training set: {anomaly_row_numbers}")
print(f"Total number of anomalies detected in the training set: {num_anomalies}")

# Prompt the user to input row numbers (comma-separated)
user_input = input("Enter row numbers to check for anomalies, separated by commas: ")

# Convert input to a list of integers
row_numbers = list(map(int, user_input.split(',')))

# Identify anomalies among the provided row numbers
anomalies_in_input = [row for row in row_numbers if train_anomalies[row]]
num_anomalies_in_input = len(anomalies_in_input)

print(f"Row numbers of anomalies from user input: {anomalies_in_input}")
print(f"Total number of anomalies detected in user-provided rows: {num_anomalies_in_input}")


# User input

# Calculate reconstruction error per feature for analysis
reconstructed_data = autoencoder.predict(data_scaled)
reconstruction_errors = np.square(data_scaled - reconstructed_data)
anomalous_columns = pd.DataFrame(reconstruction_errors, columns=features)

# Prompt user to input row indices
row_indices = input("Enter the row numbers (separated by commas) to check for highest error columns: ")
row_indices = [int(x.strip()) for x in row_indices.split(",")]

# Extract reconstruction errors only for specified rows
selected_errors = anomalous_columns.loc[row_indices]

# Identify the feature with the highest error for each specified row
max_error_columns = selected_errors.idxmax(axis=1)

# Output the results
print("Selected Rows and the Feature with Highest Error:")
for index, feature in zip(row_indices, max_error_columns):
    print(f"Row {index} - Highest error in feature: {feature}")


# Final training autoencoder with confusion matrix

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Increase training epochs or model complexity if needed
# Define more layers to capture complex patterns (optional)
encoding_dim_2 = 2  # Smaller intermediate layer

# Build a more complex autoencoder
input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="relu")(input_layer)
encoder = Dense(encoding_dim_2, activation="relu")(encoder)
decoder = Dense(encoding_dim, activation="relu")(encoder)
decoder = Dense(input_dim, activation="linear")(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)

# Compile the model
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

# Train the autoencoder on the training data with more epochs
history = autoencoder.fit(
    train_data, train_data,
    epochs=50,  # Increased epochs
    batch_size=32,
    validation_split=0.1,
    shuffle=True
)

# Calculate reconstruction error on the training data
reconstructed_train_data = autoencoder.predict(train_data)
reconstruction_error_train = np.mean(np.square(train_data - reconstructed_train_data), axis=1)

# Set a threshold for anomaly detection (e.g., 90th percentile)
threshold_train = np.percentile(reconstruction_error_train, 90)

# Identify anomalies in the training data
train_anomalies = reconstruction_error_train > threshold_train

# Calculate evaluation metrics with new threshold
accuracy_train = accuracy_score(true_labels_train, train_anomalies)
precision_train = precision_score(true_labels_train, train_anomalies)
recall_train = recall_score(true_labels_train, train_anomalies)
f1_train = f1_score(true_labels_train, train_anomalies)

# Print metrics for training set
print(f"Training Set Accuracy: {accuracy_train:.2f}")
print(f"Training Set Precision: {precision_train:.2f}")
print(f"Training Set Recall: {recall_train:.2f}")
print(f"Training Set F1 Score: {f1_train:.2f}")

# Print confusion matrix
conf_matrix = confusion_matrix(true_labels_train, train_anomalies)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix - Training Set")
plt.show()


# Plot reconstruction error for training set with updated threshold
plt.figure(figsize=(10, 6))
plt.hist(reconstruction_error_train, bins=50, color="blue", label="Reconstruction Error")
plt.axvline(threshold_train, color="red", linestyle="--", label="Threshold")
plt.title("Reconstruction Error Distribution (Training Set)")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.legend()
plt.show()