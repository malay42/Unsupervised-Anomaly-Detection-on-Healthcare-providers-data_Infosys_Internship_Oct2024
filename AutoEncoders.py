import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Show only errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.losses import MeanSquaredLogarithmicError
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("Cleaned_Healthcare Providers.csv")
columns = [
    "Number of Services", 
    "Number of Medicare Beneficiaries",
    "Number of Distinct Medicare Beneficiary/Per Day Services",
    "Average Medicare Allowed Amount",
    "Average Submitted Charge Amount",
    "Average Medicare Payment Amount",
    "Average Medicare Standardized Amount"
]
df = df[columns]

# Data scaling using StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(df)

# Train-test split for the model
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

# Train Isolation Forest on training data
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(x_train)

# Predict anomalies on the test set using Isolation Forest
iso_forest_test_labels = iso_forest.predict(x_test)
iso_forest_test_labels = np.where(iso_forest_test_labels == -1, 0, 1)  # Convert -1 to 0 for anomalies, 1 for normal

# Evaluation metrics
accuracy = accuracy_score(iso_forest_test_labels, test_predictions)
precision = precision_score(iso_forest_test_labels, test_predictions)
recall = recall_score(iso_forest_test_labels, test_predictions)
f1 = f1_score(iso_forest_test_labels, test_predictions)

print("Evaluation Metrics for Autoencoder on Test Set (with Isolation Forest Labels as Ground Truth):")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(iso_forest_test_labels, test_predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Confusion Matrix
conf_matrix = confusion_matrix(iso_forest_test_labels, test_predictions)
plt.figure(figsize=(6,6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Anomaly', 'Normal'], yticklabels=['Anomaly', 'Normal'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve and AUC
fpr, tpr, _ = roc_curve(iso_forest_test_labels, test_errors)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()

# Visualize reconstruction errors
plt.figure(figsize=(10,6))
plt.hist(train_reconstruction_errors, bins=50, color='blue', alpha=0.7)
plt.axvline(threshold, color='red', linestyle='--')
plt.title("Histogram of Reconstruction Errors")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.show()

# Training and validation loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training and Validation Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Cumulative Distribution Function (CDF) for reconstruction errors
sorted_errors = np.sort(train_reconstruction_errors)
cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

plt.figure(figsize=(10, 6))
plt.plot(sorted_errors, cdf, label="CDF of Reconstruction Errors")
plt.axvline(threshold, color='red', linestyle='--', label="Anomaly Threshold")
plt.title("Cumulative Distribution of Reconstruction Errors")
plt.xlabel("Reconstruction Error")
plt.ylabel("Cumulative Probability")
plt.legend()
plt.show()

# Count of Normal and Anomalous data points for bar chart
normal_count = train_predictions.value_counts().get(1, 0)  # Normal data points (1)
anomaly_count = train_predictions.value_counts().get(0, 0)  # Anomalous data points (0)

# Bar chart
plt.figure(figsize=(8, 6))
plt.bar(['Normal', 'Anomaly'], [normal_count, anomaly_count], color=['blue', 'red'])
plt.title('Count of Normal vs Anomalous Data Points')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()

## KDE for feature distribution for all 7 columns
plt.figure(figsize=(14, 10))
for i, column in enumerate(columns, 1):  # Iterate over all columns
    plt.subplot(3, 3, i)  # Create subplots in a 3x3 grid (9 positions, so it fits all columns)
    
    # Ensure we're using the correct test predictions for filtering
    sns.kdeplot(df[column].iloc[:len(test_predictions)][test_predictions == 0], label="Normal", fill=True, color="blue", alpha=0.3)
    sns.kdeplot(df[column].iloc[:len(test_predictions)][test_predictions == 1], label="Anomaly", fill=True, color="red", alpha=0.3)
    
    plt.title(f"Distribution of {column}")
    plt.legend()

plt.tight_layout()  # Adjust layout for better spacing
plt.show()
