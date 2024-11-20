# Outliers

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Load cleaned data
data = pd.read_csv('cleaned_data.csv')

# Columns for processing
numerical_cols = ['Number of Services', 'Number of Medicare Beneficiaries',
                  'Number of Distinct Medicare Beneficiary/Per Day Services',
                  'Average Medicare Allowed Amount', 'Average Submitted Charge Amount',
                  'Average Medicare Payment Amount', 'Average Medicare Standardized Amount']

# Scale numerical columns for outlier detection
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[numerical_cols])

# Calculate z-scores
z_scores = np.abs(scaled_data)

# Identify potential outliers
outlier_threshold = 3
outliers = np.where(z_scores > outlier_threshold)
print(f"Outliers found at indices: {np.unique(outliers[0])}")

# Visualize outliers for each column using boxplot and Z-scores
plt.figure(figsize=(16, 12))
for i, col in enumerate(numerical_cols, 1):
    z_col = z_scores[:, i-1]  # Get the Z-scores for the current column
    outlier_indices = np.where(z_col > outlier_threshold)[0]  # Indices where Z-score > threshold

    # Plot with outliers marked
    plt.subplot(3, 3, i)
    sns.boxplot(x=data[col], color='lightblue')
    plt.scatter(data.loc[outlier_indices, col], [0] * len(outlier_indices), color='red', marker='x', label='Z-score Outliers')
    plt.title(f'{col} - Z-score Outliers')
    plt.legend()

plt.tight_layout()
plt.show()

# Save the plot to a file 
# plt.savefig("outlier_detection_plots.png")

# Detect outliers using IQR method
iqr_outliers = {}
plt.figure(figsize=(16, 12))
for i, col in enumerate(numerical_cols, 1):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outlier_indices = data[(data[col] < lower_bound) | (data[col] > upper_bound)].index
    iqr_outliers[col] = outlier_indices

    # Plot with outliers marked
    plt.subplot(3, 3, i)
    sns.boxplot(x=data[col], color='lightgreen')
    plt.scatter(data.loc[outlier_indices, col], [0] * len(outlier_indices), color='red', marker='x', label='IQR Outliers')
    plt.title(f'{col} - IQR Outliers')
    plt.legend()

plt.tight_layout()
plt.show()