# Testing Autoencoders

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

# Calculate reconstruction error on the testing data
reconstructed_test_data = autoencoder.predict(test_data)
reconstruction_error = np.mean(np.square(test_data - reconstructed_test_data), axis=1)

# Set a threshold for anomaly detection (95th percentile)
threshold = np.percentile(reconstruction_error, 95)

# Identify anomalies in the testing data
anomalies = reconstruction_error > threshold

# Generate placeholder true labels for demonstration
# Assuming 5% of the test set are true anomalies
true_labels = np.zeros(len(test_data), dtype=int)
anomaly_indices = np.random.choice(len(test_data), int(0.05 * len(test_data)), replace=False)
true_labels[anomaly_indices] = 1

# Calculate evaluation metrics
accuracy = accuracy_score(true_labels, anomalies)
precision = precision_score(true_labels, anomalies)
recall = recall_score(true_labels, anomalies)
f1 = f1_score(true_labels, anomalies)

# Print metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Plot the reconstruction error
plt.figure(figsize=(10, 6))
plt.hist(reconstruction_error, bins=50, color="blue", label="Reconstruction Error")
plt.axvline(threshold, color="red", linestyle="--", label="Threshold")
plt.title("Reconstruction Error Distribution (Test Set)")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Visualize anomalies vs normal data on selected features
test_data_df = pd.DataFrame(test_data, columns=features)
test_data_df["anomaly"] = anomalies

# Separate normal and anomaly data in test set for visualization
normal_data = test_data_df[test_data_df["anomaly"] == False]
anomaly_data = test_data_df[test_data_df["anomaly"] == True]

# Define axis limits based on percentiles
x_min, x_max = normal_data['Number of Services'].quantile(0.01), normal_data['Number of Services'].quantile(0.99)
y_min, y_max = normal_data['Average Medicare Payment Amount'].quantile(0.01), normal_data['Average Medicare Payment Amount'].quantile(0.99)

plt.figure(figsize=(10, 6))

# Plot normal data in blue
plt.scatter(normal_data['Number of Services'], normal_data['Average Medicare Payment Amount'], 
            c='blue', label='Normal', alpha=0.6)
# Plot anomalies in red
plt.scatter(anomaly_data['Number of Services'], anomaly_data['Average Medicare Payment Amount'], 
            c='red', label='Anomaly', alpha=0.6)

plt.title('Anomaly Detection using Autoencoder (Test Set)')
plt.xlabel('Number of Services')
plt.ylabel('Average Medicare Payment Amount')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend()
plt.show()

# Calculate and print the number of anomalies
num_anomalies = np.sum(anomalies)
print(f"Number of anomalies detected: {num_anomalies}")


# Get the row numbers of the anomalies in the training set
anomaly_row_numbers = np.where(anomalies)[0]  # Row numbers where train_anomalies is True
num_anomalies = len(anomaly_row_numbers)  # Count of anomalies

print(f"Row numbers of anomalies in the training set: {anomaly_row_numbers}")
print(f"Total number of anomalies detected in the training set: {num_anomalies}")

# Prompt the user to input row numbers (comma-separated)
user_input = input("Enter row numbers to check for anomalies, separated by commas: ")

# Convert input to a list of integers
row_numbers = list(map(int, user_input.split(',')))

# Identify anomalies among the provided row numbers
anomalies_in_input = [row for row in row_numbers if anomalies[row]]
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


# Final Testing autoencoder with Confusion Matrix

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Increase training epochs or add a second encoding layer for complexity
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

# Train the autoencoder on the training data with increased epochs
history = autoencoder.fit(
    train_data, train_data,
    epochs=50,  # Increased epochs
    batch_size=32,
    validation_split=0.1,
    shuffle=True
)

# Calculate reconstruction error on the test data
reconstructed_test_data = autoencoder.predict(test_data)
reconstruction_error = np.mean(np.square(test_data - reconstructed_test_data), axis=1)

# Set a threshold for anomaly detection (e.g., 90th percentile)
threshold = np.percentile(reconstruction_error, 90)

# Identify anomalies in the test data
anomalies = reconstruction_error > threshold

# Calculate evaluation metrics
accuracy = accuracy_score(true_labels, anomalies)
precision = precision_score(true_labels, anomalies)
recall = recall_score(true_labels, anomalies)
f1 = f1_score(true_labels, anomalies)

# Print metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Display confusion matrix
conf_matrix = confusion_matrix(true_labels, anomalies)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix - Test Set")
plt.show()

# Plot the reconstruction error distribution with updated threshold
plt.figure(figsize=(10, 6))
plt.hist(reconstruction_error, bins=50, color="blue", label="Reconstruction Error")
plt.axvline(threshold, color="red", linestyle="--", label="Threshold")
plt.title("Reconstruction Error Distribution (Test Set)")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.legend()
plt.show()


# Visualize anomalies vs normal data on selected features in the test set
test_data_df = pd.DataFrame(test_data, columns=features)
test_data_df["anomaly"] = anomalies

# Separate normal and anomaly data for visualization
normal_data = test_data_df[test_data_df["anomaly"] == False]
anomaly_data = test_data_df[test_data_df["anomaly"] == True]

# Define axis limits based on percentiles
x_min, x_max = normal_data['Number of Services'].quantile(0.01), normal_data['Number of Services'].quantile(0.99)
y_min, y_max = normal_data['Average Medicare Payment Amount'].quantile(0.01), normal_data['Average Medicare Payment Amount'].quantile(0.99)

plt.figure(figsize=(10, 6))

# Plot normal data in blue
plt.scatter(normal_data['Number of Services'], normal_data['Average Medicare Payment Amount'], 
            c='blue', label='Normal', alpha=0.6)
# Plot anomalies in red
plt.scatter(anomaly_data['Number of Services'], anomaly_data['Average Medicare Payment Amount'], 
            c='red', label='Anomaly', alpha=0.6)

plt.title('Anomaly Detection using Autoencoder (Test Set)')
plt.xlabel('Number of Services')
plt.ylabel('Average Medicare Payment Amount')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend()
plt.show()