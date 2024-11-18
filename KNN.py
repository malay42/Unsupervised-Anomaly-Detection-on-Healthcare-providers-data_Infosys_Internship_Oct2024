# Fit the KNN model on the entire dataset
knn = NearestNeighbors(n_neighbors=5)
knn.fit(final_dataset[numeric_columns])

# Calculate distances and indices of the nearest neighbors for the entire dataset
distances, indices = knn.kneighbors(final_dataset[numeric_columns])

# Use the distance to the nearest neighbor as the anomaly score
anomaly_scores = distances[:, -1]  # The last column corresponds to the distance to the farthest neighbor

# Set a threshold for anomaly detection
threshold = anomaly_scores.mean() + anomaly_scores.std()

# Classify as anomaly if the score is above the threshold
final_dataset['anomaly_knn'] = (anomaly_scores > threshold).astype(int)

# Separate normal data and anomalies
anomalies_knn = final_dataset[final_dataset['anomaly_knn'] == 1]
normal_data_knn = final_dataset[final_dataset['anomaly_knn'] == 0]

# Display anomalies
print("Anomalies detected by KNN:")
print(anomalies_knn.head())
print(f"Total number of anomalies detected by KNN: {anomalies_knn.shape[0]}")
# Fit the KNN model on the training set
knn = NearestNeighbors(n_neighbors=5)
knn.fit(X_train)

# Calculate distances and indices of the nearest neighbors for the test set
distances, indices = knn.kneighbors(X_test)

# Use the distance to the nearest neighbor as the anomaly score
anomaly_scores = distances[:, -1]  # The last column corresponds to the distance to the farthest neighbor

# Set a threshold for anomaly detection
threshold = anomaly_scores.mean() + anomaly_scores.std()

# Create a separate DataFrame for the test set with anomaly labels
test_set_with_anomalies = X_test.copy()
test_set_with_anomalies['anomaly_knn'] = (anomaly_scores > threshold).astype(int)

# Separate normal data and anomalies
anomalies_knn = test_set_with_anomalies[test_set_with_anomalies['anomaly_knn'] == 1]
normal_data_knn = test_set_with_anomalies[test_set_with_anomalies['anomaly_knn'] == 0]

# Display anomalies
print("Anomalies detected by KNN in test set:")
print(anomalies_knn.head())
print(f"Total number of anomalies detected by KNN in test set: {anomalies_knn.shape[0]}")
# Visualization for the entire dataset
plt.figure(figsize=(10, 6))

# Plot normal data
plt.scatter(
    final_dataset[final_dataset['anomaly_knn'] == 0]['Number of Services'],
    final_dataset[final_dataset['anomaly_knn'] == 0]['Average Medicare Payment Amount'],
    color='blue', label='Normal Data', alpha=0.5
)

# Plot anomalies
plt.scatter(
    final_dataset[final_dataset['anomaly_knn'] == 1]['Number of Services'],
    final_dataset[final_dataset['anomaly_knn'] == 1]['Average Medicare Payment Amount'],
    color='red', label='Anomalies', alpha=0.7
)

# Adding labels and title
plt.xlabel('Number of Services')
plt.ylabel('Average Medicare Payment Amount')
plt.title('Anomaly Detection using KNN')
plt.legend()
plt.grid()
plt.show()

# Optional: Histogram of Anomaly Scores
plt.figure(figsize=(10, 6))
plt.hist(anomaly_scores, bins=50, color='gray', alpha=0.7, edgecolor='black')
plt.axvline(x=threshold, color='red', linestyle='--', label='Threshold')
plt.title('Histogram of Anomaly Scores')
plt.xlabel('Anomaly Score (Distance to Farthest Neighbor)')
plt.ylabel('Frequency')
plt.legend()
plt.show()

