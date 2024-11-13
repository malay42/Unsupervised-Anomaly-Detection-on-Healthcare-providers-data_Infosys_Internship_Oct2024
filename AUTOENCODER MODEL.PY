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



df = data_dropped



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
plt.savefig('Autoencoder distrubution.png')  
plt.show()

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix at 94th Percentile Threshold autoencoder")
plt.savefig('Confusion Matrix at 94th Percentile Threshold autoencoder.png')  
plt.show()


def check_anomaly(row_num, reconstruction_error, optimal_threshold):
    if reconstruction_error[row_num] > optimal_threshold:
        return "Anomaly"
    else:
        return "Normal"


row_num = int(input("Enter the row number to check for anomaly (0-based index): "))


if row_num < 0 or row_num >= len(df):
    print("Invalid row number. Please enter a number between 0 and", len(df)-1)
else:

    result = check_anomaly(row_num, reconstruction_error, optimal_threshold)
    print(f"Row {row_num} is: {result}")
  
  
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds_roc = roc_curve(iso_labels == -1, reconstruction_error)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.savefig('ROC CURVE AUTOECNODER.png')  
plt.show()


plt.figure(figsize=(12, 8))
for i, column in enumerate(numeric_columns[:4], 1): 
    plt.subplot(2, 2, i)
    sns.kdeplot(df[column][predicted_anomalies == 0], label="Normal", fill=True, color="blue", alpha=0.3)
    sns.kdeplot(df[column][predicted_anomalies == 1], label="Anomaly", fill=True, color="red", alpha=0.3)
    plt.title(f"Distribution of {column}")
    plt.legend()
plt.tight_layout()
plt.savefig('Autoencoder feature distrubution.png')  
plt.show()
  