import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.losses import MeanSquaredLogarithmicError
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("Cleaned_Healthcare Providers.csv")
columns = ["Number of Services", 
           "Number of Medicare Beneficiaries",
           "Number of Distinct Medicare Beneficiary/Per Day Services",
           "Average Medicare Allowed Amount",
           "Average Submitted Charge Amount",
           "Average Medicare Payment Amount",
           "Average Medicare Standardized Amount"]
df = df[columns]

# Data scaling using StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(df)

# Train-test split for the model (since it's unsupervised, we just use it for validation later)
x_train, x_test = train_test_split(x_scaled, test_size=0.2, random_state=42)

# Define Autoencoder Model
class AutoEncoder(Model):
    def __init__(self, output_units, code_size=8):
        super().__init__()
        self.encoder = Sequential([
            Dense(256, activation='relu'),
            Dropout(0.1),
            Dense(128, activation='relu'),
            Dropout(0.1),
            Dense(64, activation='relu'),
            Dropout(0.1),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(code_size, activation='relu')
        ])
        self.decoder = Sequential([
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(64, activation='relu'),
            Dropout(0.1),
            Dense(128, activation='relu'),
            Dropout(0.1),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(output_units, activation='sigmoid')
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

# Create and compile the autoencoder model
model = AutoEncoder(output_units=x_train.shape[1])
model.compile(loss=MeanSquaredLogarithmicError(), optimizer=Adam())

# Train the model
history = model.fit(x_train, x_train, epochs=20, batch_size=32, validation_data=(x_test, x_test))
# Function to find threshold
def find_threshold(model, data):
    reconstructions = model.predict(data)
    reconstruction_errors = tf.keras.losses.msle(reconstructions, data).numpy()
    threshold = np.mean(reconstruction_errors) + np.std(reconstruction_errors)
    return threshold, reconstruction_errors

# Calculate the threshold based on the training set
threshold, train_reconstruction_errors = find_threshold(model, x_train)
print(f"Threshold: {threshold}")

# Get anomaly predictions based on the threshold
def get_predictions(model, data, threshold):
    predictions = model.predict(data)
    errors = tf.keras.losses.msle(predictions, data).numpy()
    anomaly_mask = pd.Series(errors) > threshold
    preds = anomaly_mask.map(lambda x: 0.0 if x else 1.0)
    return preds, errors

# Detect anomalies in training and test sets
train_predictions, train_errors = get_predictions(model, x_train, threshold)
test_predictions, test_errors = get_predictions(model, x_test, threshold)

# Count anomalies in the entire dataset
num_train_anomalies = train_predictions.value_counts().get(0, 0)
num_test_anomalies = test_predictions.value_counts().get(0, 0)
total_anomalies = num_train_anomalies + num_test_anomalies
print(f"Number of Anomalies in Training Set: {num_train_anomalies}")
print(f"Number of Anomalies in Test Set: {num_test_anomalies}")
print(f"Total Anomalies Detected: {total_anomalies}")

# Visualize the reconstruction errors
plt.figure(figsize=(10,6))
plt.hist(train_reconstruction_errors, bins=50, color='blue', alpha=0.7)
plt.axvline(threshold, color='red', linestyle='--')
plt.title("Histogram of Reconstruction Errors")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.show()

# Visualize the anomaly predictions for training set
plt.figure(figsize=(10,6))
sns.scatterplot(x=np.arange(len(train_errors)), y=train_errors, color=['red' if anomaly else 'blue' for anomaly in train_predictions])
plt.title("Anomaly Detection Using Reconstruction Errors (Training Set)")
plt.xlabel("Index")
plt.ylabel("Reconstruction Error")
plt.show()

## plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training and Validation Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# Calculate Cumulative Distribution Function (CDF) for reconstruction errors
sorted_errors = np.sort(train_reconstruction_errors)
cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

# Plot CDF with threshold line
plt.figure(figsize=(10, 6))
plt.plot(sorted_errors, cdf, label="CDF of Reconstruction Errors")
plt.axvline(threshold, color='red', linestyle='--', label="Anomaly Threshold")
plt.title("Cumulative Distribution of Reconstruction Errors")
plt.xlabel("Reconstruction Error")
plt.ylabel("Cumulative Probability")
plt.legend()
plt.show()

# Count normal and anomalous data points
normal_count = train_predictions.value_counts().get(1, 0)  # Normal data points (1)
anomaly_count = train_predictions.value_counts().get(0, 0)  # Anomalous data points (0)

# Data for the bar chart
categories = ['Normal', 'Anomaly']
counts = [normal_count, anomaly_count]

# Create the bar chart
plt.figure(figsize=(8, 6))
plt.bar(categories, counts, color=['blue', 'red'])
plt.title('Count of Normal vs Anomalous Data Points')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()
