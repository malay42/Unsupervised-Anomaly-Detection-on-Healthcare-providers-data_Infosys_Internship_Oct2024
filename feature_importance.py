import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np

# Load the dataset
data = pd.read_csv('cleaned2_encoded.csv')

# numeric columns
numeric_data = data.select_dtypes(include=[np.number])

# Isolation Forest model
isolation_forest = IsolationForest(contamination=0.05, random_state=42)
isolation_forest.fit(numeric_data)

# anomaly scores
anomaly_scores = isolation_forest.decision_function(numeric_data)

# feature importance by observing change in anomaly scores when features are shuffled
importances = {}
for feature in numeric_data.columns:
    # Shuffle the current feature's values
    shuffled_data = numeric_data.copy()
    shuffled_data[feature] = np.random.permutation(shuffled_data[feature])
    
    # anomaly scores with shuffled feature
    shuffled_scores = isolation_forest.decision_function(shuffled_data)
    
    # importance by calculating the change in scores
    importance = np.mean(np.abs(anomaly_scores - shuffled_scores))
    importances[feature] = importance

# Convert to DataFrame for easy viewing and sorting
feature_importances = pd.DataFrame(list(importances.items()), columns=['Feature', 'Importance'])
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# top important features
print("Top important features:")
print(feature_importances.head(10))
