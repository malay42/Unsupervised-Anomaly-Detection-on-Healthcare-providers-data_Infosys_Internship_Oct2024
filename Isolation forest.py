from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Apply Isolation Forest for anomaly detection
isolation_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
final_dataset['anomaly'] = isolation_forest.fit_predict(final_dataset[numeric_columns])

# Count anomalies (-1 indicates anomaly, 1 indicates normal)
anomalies = final_dataset[final_dataset['anomaly'] == -1]
print("Anomalies detected by Isolation Forest:")
print(anomalies.head())
print(f"Total number of anomalies detected by Isolation Forest: {anomalies.shape[0]}")

# Scatter plot of Average Medicare Payment Amount vs Number of Medicare Beneficiaries
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    final_dataset['Average Medicare Payment Amount'], 
    final_dataset['Number of Medicare Beneficiaries'],
    c=final_dataset['anomaly'], 
    cmap='coolwarm'
)
plt.colorbar(scatter, label='Anomaly Status (1: Normal, -1: Anomaly)')
plt.xlabel('Average Medicare Payment Amount')
plt.ylabel('Number of Medicare Beneficiaries')
plt.title('Scatter Plot of Average Medicare Payment Amount vs Number of Medicare Beneficiaries (Isolation Forest)')
plt.axhline(0, color='gray', lw=0.5)
plt.axvline(0, color='gray', lw=0.5)
plt.grid()
plt.show()
