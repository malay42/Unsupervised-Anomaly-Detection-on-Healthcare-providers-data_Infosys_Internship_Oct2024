import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns


file_path = "final.csv"
df = pd.read_csv(file_path)


numeric_columns = [
    'Number of Services', 'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services',
    'Average Medicare Allowed Amount', 'Average Submitted Charge Amount',
    'Average Medicare Payment Amount', 'Average Medicare Standardized Amount'
]


scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])


X = df[numeric_columns].values


input_dim = X.shape[1]
encoding_dim = 4

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="relu")(input_layer)
encoder = Dropout(0.2)(encoder)
decoder = Dense(input_dim, activation="linear")(encoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')


history = autoencoder.fit(X, X, epochs=20, batch_size=32, shuffle=True, validation_split=0.2, verbose=1)


X_pred = autoencoder.predict(X)
reconstruction_error = np.mean(np.square(X - X_pred), axis=1)


iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_labels = iso_forest.fit_predict(reconstruction_error.reshape(-1, 1))


best_threshold = None
best_f1 = 0
thresholds = range(80, 100, 2)


scores = []

for percentile in thresholds:
    anomaly_threshold = np.percentile(reconstruction_error, percentile)
    predicted_anomalies = (reconstruction_error > anomaly_threshold).astype(int)

    # Calculate accuracy metrics for the current threshold
    accuracy = accuracy_score(iso_labels == -1, predicted_anomalies)
    precision = precision_score(iso_labels == -1, predicted_anomalies)
    recall = recall_score(iso_labels == -1, predicted_anomalies)
    f1 = f1_score(iso_labels == -1, predicted_anomalies)

    scores.append({'Threshold Percentile': percentile, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1})


    if f1 > best_f1:
        best_f1 = f1
        best_threshold = anomaly_threshold


scores_df = pd.DataFrame(scores)
print(scores_df)


optimal_threshold = np.percentile(reconstruction_error, 94)


predicted_anomalies = (reconstruction_error > optimal_threshold).astype(int)


anomalies = df[predicted_anomalies == 1]
print("Detected Anomalies:")
print(anomalies)


accuracy = accuracy_score(iso_labels == -1, predicted_anomalies)
precision = precision_score(iso_labels == -1, predicted_anomalies)
recall = recall_score(iso_labels == -1, predicted_anomalies)
f1 = f1_score(iso_labels == -1, predicted_anomalies)

print(f"Optimal Threshold (94th Percentile): {optimal_threshold:.2f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


cm = confusion_matrix(iso_labels == -1, predicted_anomalies)
print(f"Confusion Matrix:\n{cm}")

# Visualization


plt.figure(figsize=(15, 6))

# Subplot 1: Training Loss Over Epochs
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss Over Epochs")

# Subplot 2: Reconstruction Error Distribution with Anomaly Threshold
plt.subplot(1, 2, 2)
plt.hist(reconstruction_error, bins=50, color="skyblue", alpha=0.7, label="Reconstruction Errors")
plt.axvline(optimal_threshold, color='red', linestyle='dashed', linewidth=2, label="Anomaly Threshold")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.legend()
plt.title("Reconstruction Error Distribution with Anomaly Threshold")


plt.tight_layout()
plt.show()

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix at 94th Percentile Threshold")
plt.show()