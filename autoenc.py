
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# %matplotlib inline


from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('data_encoded.csv')
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)


iso_forest = IsolationForest(contamination=0.06, max_features=0.7, max_samples=0.8, n_estimators=100, random_state=42)
df['anomaly'] = iso_forest.fit_predict(df)
df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})
df['anomaly'].value_counts()

X = df.drop(columns=['anomaly'])
y = df['anomaly']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled=scaler.transform(X)

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(64, activation='relu'),
    Dense(128, activation='relu'),
    Dense(X_scaled.shape[1], activation='sigmoid')
])

model.compile(optimizer='adam', loss='mse')

history = model.fit(X_scaled, X_scaled, epochs=5, batch_size=32, validation_data=(X_scaled,X_scaled))

X_pred = autoencoder.predict(X_scaled)
reconstruction_error = np.mean(np.square(X_scaled - X_pred), axis=1)

threshold = np.percentile(reconstruction_error, 95)
y_pred = (reconstruction_error > threshold).astype(int)

np.count_nonzero(y_pred == 1)

conf_matrix = confusion_matrix(y, y_pred)
ConfusionMatrixDisplay(conf_matrix).plot(cmap="Blues")
plt.title("Confusion Matrix - Anomaly Detection with Autoencoder")
plt.show()

accuracy = accuracy_score(y, y_pred)
print(f"Accuracy: {accuracy:.4f}")

precision = precision_score(y, y_pred, pos_label=1)
recall = recall_score(y, y_pred, pos_label=1)
f1 = f1_score(y, y_pred, pos_label=1)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

print("\nClassification Report:\n", classification_report(y, y_pred))

