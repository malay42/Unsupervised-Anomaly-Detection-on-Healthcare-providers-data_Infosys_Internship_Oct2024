# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns

from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers, losses

# Function Definitions
def load_data(file_path):
    """Load and preprocess the dataset."""
    df = pd.read_csv(file_path)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df, df_scaled

def detect_anomalies(df, contamination=0.06):
    """Apply Isolation Forest to detect anomalies."""
    iso_forest = IsolationForest(
        contamination=contamination,
        max_features=0.7,
        max_samples=0.8,
        n_estimators=100,
        random_state=42,
    )
    df['anomaly'] = iso_forest.fit_predict(df)
    df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})  # Map anomalies to 1
    return df

def split_data(df):
    """Split the dataset into training and testing sets."""
    labels = df.iloc[:, -1].values
    data = df.iloc[:, :-1].values
    return train_test_split(data, labels, test_size=0.2, random_state=21)

def preprocess_pipeline(train_data, test_data):
    """Apply normalization and scaling to the data."""
    pipeline = Pipeline([
        ('normalizer', Normalizer()),
        ('scaler', MinMaxScaler())
    ])
    pipeline.fit(train_data)
    return pipeline.transform(train_data), pipeline.transform(test_data)

class AnomalyDetector(Model):
    def __init__(self, input_dim):
        super(AnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(8, activation="relu")
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(16, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(input_dim, activation="sigmoid")
        ])

    def call(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)

# Plot results
def plot_results(history):
    """Plot training and validation loss over epochs."""
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.title("Model Loss")
    plt.show()

def plot_comparison(normal_data, anomalous_data, x_label, y_label):
    """Scatter plot for normal and anomalous data."""
    plt.scatter(normal_data[:, 0], normal_data[:, 1], color='blue', alpha=0.5, label='Normal', s=50)
    plt.scatter(anomalous_data[:, 0], anomalous_data[:, 1], color='red', alpha=0.5, label='Anomalous', s=50)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title("Comparison of Normal and Anomalous Data")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def plot_reconstruction(original, reconstructed, label):
    """Plot reconstructed and original data."""
    plt.plot(original[0], 'b', label="Original")
    plt.plot(reconstructed[0], 'r', label="Reconstructed")
    plt.fill_between(np.arange(len(original[0])), original[0], reconstructed[0], color='lightcoral', alpha=0.5)
    plt.title(f"Reconstruction Error ({label})")
    plt.legend()
    plt.show()
    
def plot_train_loss(loss):
    plt.hist(loss[None,:], bins=50)
    plt.xlabel("Train loss")
    plt.ylabel("No of examples")
    plt.show()
    

def plot_distribution(train_loss, test_loss, threshold):
    """Plot the distribution of reconstruction loss."""
    plt.hist(train_loss, bins=50, density=True, alpha=0.6, color='green', label="Clean")
    plt.hist(test_loss, bins=50, density=True, alpha=0.6, color='red', label="Anomalous")
    plt.axvline(threshold, color='black', linestyle='dashed', linewidth=2, label="Threshold")
    plt.legend()
    plt.title("Reconstruction Loss Distribution")
    plt.show()

# Calculate and print statistics
def print_stats(predictions, labels):
    """Print model evaluation metrics."""
    print(f"Accuracy: {accuracy_score(labels, predictions):.2f}")
    print(f"Precision: {precision_score(labels, predictions):.2f}")
    print(f"Recall: {recall_score(labels, predictions):.2f}")
    
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Anomaly', 'Normal'], yticklabels=['Anomaly', 'Normal'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    file_path = 'data_encoded.csv'
    df, data_scaled = load_data(file_path)

    df = detect_anomalies(df)
    train_data, test_data, train_labels, test_labels = split_data(df)
    train_data, test_data = preprocess_pipeline(train_data, test_data)

    # Separate normal and anomalous data
    train_labels = train_labels.astype(bool)
    test_labels = test_labels.astype(bool)
    normal_train_data = train_data[train_labels]
    anomalous_train_data = train_data[~train_labels]
    normal_test_data = test_data[test_labels]
    anomalous_test_data = test_data[~test_labels]

    # Plot normal vs anomalous data
    x_label = df.columns[6]
    y_label = df.columns[11]
    plot_comparison(normal_train_data, anomalous_train_data, x_label, y_label)

    # Initialize and train autoencoder
    autoencoder = AnomalyDetector(input_dim=train_data.shape[1])
    autoencoder.compile(optimizer='adam', loss='mae')
    history = autoencoder.fit(
        normal_train_data, normal_train_data,
        epochs=25,
        batch_size=250,
        validation_data=(test_data, test_data),
        shuffle=True
    )

    plot_results(history)

    # Evaluate reconstruction loss
    reconstructed_normal = autoencoder(normal_test_data).numpy()
    reconstructed_anomalous = autoencoder(anomalous_test_data).numpy()
    plot_reconstruction(normal_test_data, reconstructed_normal, "Normal")
    plot_reconstruction(anomalous_test_data, reconstructed_anomalous, "Anomalous")

    reconstructions = autoencoder.predict(normal_train_data)
    train_loss = tf.keras.losses.mae(reconstructions, normal_train_data)
    
    # Plot train loss
    plot_train_loss(train_loss)

    threshold = np.mean(train_loss) + 2 * np.std(train_loss)
    print("Threshold : ", threshold)

    test_reconstructions = autoencoder.predict(anomalous_test_data)
    test_loss = tf.keras.losses.mae(test_reconstructions, anomalous_test_data)
    
    plot_train_loss(test_loss)

    plot_distribution(train_loss, test_loss, threshold)

    # Predict anomalies and evaluate
    reconstructions = autoencoder(test_data)
    loss = tf.keras.losses.mae(reconstructions, test_data)
    preds = tf.math.less(loss, threshold)
    print_stats(preds, test_labels)
