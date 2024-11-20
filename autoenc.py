import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l1, l2

# Load and scale the dataset
df = pd.read_csv("data_encoded.csv")
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

# Detect anomalies using Isolation Forest
iso_forest = IsolationForest(
    contamination=0.06, max_features=0.7, max_samples=0.8, n_estimators=100, random_state=42
)
df["anomaly"] = iso_forest.fit_predict(df)
df["anomaly"] = df["anomaly"].map({1: 0, -1: 1})
print(df["anomaly"].value_counts())

# Prepare features and labels
features, anomalies = df.drop("anomaly", axis=1).values, df["anomaly"].values


def plot_tsne(features, anomalies, dimensions=2):
    """Plots t-SNE scatter plot for anomalies and normal data."""
    embedded_features = TSNE(n_components=dimensions, random_state=42).fit_transform(features)
    plt.figure(figsize=(8, 8))
    plt.scatter(*zip(*embedded_features[anomalies == 1]), c="r", s=2, alpha=0.7, label="Anomaly")
    plt.scatter(*zip(*embedded_features[anomalies == 0]), c="g", s=2, alpha=0.3, label="Normal")
    plt.legend(loc="best")
    plt.savefig("tsne_scatter.png")
    plt.show()


plot_tsne(features, anomalies)

# Split the dataset into train, validation, and test sets
X_normal = df[df["anomaly"] == 0]
X_anomalous = df[df["anomaly"] == 1]
X_train, X_val_normal = train_test_split(X_normal, test_size=0.3, random_state=42)
X_val_anom, X_test_anom = train_test_split(X_anomalous, test_size=0.35, random_state=42)
X_val = pd.concat([X_val_normal, X_val_anom]).sample(frac=1, random_state=42)
X_test = pd.concat([X_normal.sample(12000, random_state=42), X_test_anom]).sample(frac=1, random_state=42)

y_val, y_test = X_val["anomaly"], X_test["anomaly"]
X_train, X_val, X_test = (
    X_train.drop("anomaly", axis=1),
    X_val.drop("anomaly", axis=1),
    X_test.drop("anomaly", axis=1),
)

print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")

# Normalize and scale data
pipeline = Pipeline([("normalizer", Normalizer()), ("scaler", MinMaxScaler())])
X_train = pipeline.fit_transform(X_train)
X_val = pipeline.transform(X_val)
X_test = pipeline.transform(X_test)

# Build and compile the autoencoder model
input_dim = X_train.shape[1]
model = Sequential([
    Dense(60, activation="relu", input_shape=(input_dim,), kernel_regularizer=l2(0.01)),
    Dropout(0.1),
    Dense(15, activation="relu", kernel_regularizer=l2(0.01)),
    Dropout(0.1),
    Dense(10, activation="relu", kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dense(2, activation="relu", kernel_regularizer=l1(0.005)),
    Dropout(0.1),
    Dense(10, activation="relu", kernel_regularizer=l2(0.01)),
    Dense(input_dim, activation="relu", kernel_regularizer=l2(0.01)),
])

model.compile(optimizer="adam", loss="mse")
model.summary()

# Callbacks for training
early_stopping = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(
    filepath="best_autoencoder_weights.h5", monitor="val_loss", save_best_only=True
)

# Train the model
model.fit(
    X_train, X_train,
    epochs=25, batch_size=32, shuffle=True,
    validation_data=(X_val, X_val),
    callbacks=[early_stopping, model_checkpoint],
    verbose=1,
)

# Evaluate reconstruction error for test data
X_pred = model.predict(X_test)
reconstruction_error = np.mean(np.square(X_test - X_pred), axis=1)
threshold = np.percentile(reconstruction_error, 95)
predicted_anomalies = (reconstruction_error > threshold).astype(int)

# Evaluate model performance 
def print_metrics(y_true, y_pred):
    """Prints classification metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1)
    recall = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_true, y_pred))

print_metrics(y_test, predicted_anomalies)

# Evaluate performance for entire dataset
X = df.drop('anomaly', axis=1)
X_transformed = pipeline.transform(X)
pred = model.predict(X_transformed)
reconstruction_error = np.mean(np.square(X_transformed - pred), axis=1)
threshold = np.percentile(reconstruction_error, 95)

# Detect anomalies 
ae_anomaly = (reconstruction_error > threshold).astype(int)
unique_values, counts = np.unique(ae_anomaly, return_counts=True)
print(unique_values, counts, threshold)

# Print performance metrics
print_metrics(df['anomaly'], ae_anomaly)

# Plot the distribution of reconstruction errors
plt.figure(figsize=(8, 6))
plt.hist(reconstruction_error, bins=50, color="skyblue", alpha=0.7, label="Reconstruction Errors")
plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label="Anomaly Threshold")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.legend()
plt.title("Reconstruction Error Distribution with Anomaly Threshold")

# distribution of reconstruction errors for anomalous and normal data as separate
clean_errors = reconstruction_error[df['anomaly'] == 0]
anomaly_errors = reconstruction_error[df['anomaly'] == 1]
fig, ax = plt.subplots(figsize=(6, 6))
ax.hist(clean_errors, bins=50, density=True, label="Normal", alpha=0.6, color="green")
ax.hist(anomaly_errors, bins=50, density=True, label="Anomaly", alpha=0.6, color="red")
plt.axvline(threshold, color='black', linestyle='dashed', linewidth=2, label="Anomaly Threshold")
plt.title("(Normalized) Distribution of the Reconstruction Loss")
plt.legend()
plt.show()

# Define numerical columns for visualization
df['ae_anomaly'] = ae_anomaly

num_columns = [
    'Number of Services', 'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services',
    'Average Medicare Allowed Amount', 'Average Submitted Charge Amount',
    'Average Medicare Payment Amount', 'Average Medicare Standardized Amount'
]

plt.figure(figsize=(12, 10))
for i, column in enumerate(num_columns, 1):
    plt.subplot(4, 2, i)
    sns.kdeplot(df[column][df['anomaly'] == 0], label="Normal", fill=True, color="blue", alpha=0.3)
    sns.kdeplot(df[column][df['anomaly'] == 1], label="Anomaly", fill=True, color="red", alpha=0.3)
    plt.title(f"Distribution of {column}")
    plt.legend()
plt.tight_layout()
plt.savefig('Autoencoder_feature_distribution.png')
plt.show()

# Plot the KDE for a specific feature ('Average Submitted Charge Amount')
plt.figure(figsize=(6, 5))
sns.kdeplot(df['Average Submitted Charge Amount'][df['ae_anomaly'] == 0], label="Normal", fill=True, color="blue", alpha=0.3)
sns.kdeplot(df['Average Submitted Charge Amount'][df['ae_anomaly'] == 1], label="Anomaly", fill=True, color="red", alpha=0.3)
plt.ylim(0.00, 0.01)
plt.title(f"Distribution of Average Submitted Charge Amount")
plt.legend()
plt.tight_layout()
plt.show()
