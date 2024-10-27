# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

# Load the data
file_path = "C:/Users/shiva/Desktop/Project 1/Cleaned_Healthcare_Providers.csv"
df = pd.read_csv(file_path)

# Define numeric columns
numeric_columns = [
    'Number of Services', 'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services',
    'Average Medicare Allowed Amount', 'Average Submitted Charge Amount',
    'Average Medicare Payment Amount', 'Average Medicare Standardized Amount'
]

# Convert numeric columns to float and remove commas
df[numeric_columns] = df[numeric_columns].replace({',': ''}, regex=True).astype(float)

# Function to plot histograms of numeric columns
def plot_histograms(data, title):
    data[numeric_columns].hist(bins=30, figsize=(12, 10), layout=(3, 3))
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Function to visualize correlation heatmap
def plot_correlation_heatmap(df, title):
    plt.figure(figsize=(10, 8))
    corr = df[numeric_columns].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(title)
    plt.show()

# Plot original data histograms and correlation heatmap
plot_histograms(df, "Histograms of Original Data")
plot_correlation_heatmap(df, "Correlation Heatmap of Original Data")

# Calculate IQR bounds
Q1 = df[numeric_columns].quantile(0.25)
Q3 = df[numeric_columns].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Outlier Detection & Removal Functions

# 1. IQR Method
def remove_outliers_iqr(df):
    print("\nBefore IQR Outlier Removal:")
    print(df[numeric_columns].head())
    print("Data Size:", df.shape)
    plot_histograms(df, "Histograms Before IQR Outlier Removal")
    df_iqr_removed = df[~((df[numeric_columns] < lower_bound) | 
                          (df[numeric_columns] > upper_bound)).any(axis=1)]
    print("\nAfter IQR Outlier Removal:")
    print(df_iqr_removed[numeric_columns].head())
    print("Data Size:", df_iqr_removed.shape)
    plot_histograms(df_iqr_removed, "Histograms After IQR Outlier Removal")
    return df_iqr_removed

# 2. Quantile-based Z-Score Equivalent
def remove_outliers_quantile_zscore(df):
    print("\nBefore Quantile Z-Score Outlier Removal:")
    print(df[numeric_columns].head())
    print("Data Size:", df.shape)
    plot_histograms(df, "Histograms Before Quantile Z-Score Outlier Removal")
    z_lower_bound = Q1 - 1.5 * IQR
    z_upper_bound = Q3 + 1.5 * IQR
    df_zscore_removed = df[~((df[numeric_columns] < z_lower_bound) | 
                             (df[numeric_columns] > z_upper_bound)).any(axis=1)]
    print("\nAfter Quantile Z-Score Outlier Removal:")
    print(df_zscore_removed[numeric_columns].head())
    print("Data Size:", df_zscore_removed.shape)
    plot_histograms(df_zscore_removed, "Histograms After Quantile Z-Score Outlier Removal")
    return df_zscore_removed

# 3. Quantile-based Standard Deviation Equivalent
def remove_outliers_quantile_std(df):
    print("\nBefore Quantile Standard Deviation Outlier Removal:")
    print(df[numeric_columns].head())
    print("Data Size:", df.shape)
    plot_histograms(df, "Histograms Before Quantile Standard Deviation Outlier Removal")
    df_std_removed = df[~((df[numeric_columns] < lower_bound) | 
                          (df[numeric_columns] > upper_bound)).any(axis=1)]
    print("\nAfter Quantile Standard Deviation Outlier Removal:")
    print(df_std_removed[numeric_columns].head())
    print("Data Size:", df_std_removed.shape)
    plot_histograms(df_std_removed, "Histograms After Quantile Standard Deviation Outlier Removal")
    return df_std_removed

# 4. KNN with Quantile Threshold
def remove_outliers_knn(df, n_neighbors=5):
    print("\nBefore KNN Outlier Removal:")
    print(df[numeric_columns].head())
    print("Data Size:", df.shape)
    plot_histograms(df, "Histograms Before KNN Outlier Removal")
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(df[numeric_columns])
    distances, _ = knn.kneighbors(df[numeric_columns])
    threshold = upper_bound.mean() + 1.5 * IQR.mean()  # Quantile-based threshold
    df_knn_removed = df[distances[:, -1] < threshold]
    print("\nAfter KNN Outlier Removal:")
    print(df_knn_removed[numeric_columns].head())
    print("Data Size:", df_knn_removed.shape)
    plot_histograms(df_knn_removed, "Histograms After KNN Outlier Removal")
    return df_knn_removed

# 5. Local Outlier Factor (LOF) with Quantile Threshold
def remove_outliers_lof(df, contamination=0.1):
    print("\nBefore LOF Outlier Removal:")
    print(df[numeric_columns].head())
    print("Data Size:", df.shape)
    plot_histograms(df, "Histograms Before LOF Outlier Removal")
    lof = LocalOutlierFactor(contamination=contamination)
    y_pred = lof.fit_predict(df[numeric_columns])
    df_lof_removed = df[y_pred != -1]
    print("\nAfter LOF Outlier Removal:")
    print(df_lof_removed[numeric_columns].head())
    print("Data Size:", df_lof_removed.shape)
    plot_histograms(df_lof_removed, "Histograms After LOF Outlier Removal")
    return df_lof_removed

# 6. Robust Method (25th and 75th Percentiles)
def remove_outliers_robust(df):
    print("\nBefore Robust Outlier Removal:")
    print(df[numeric_columns].head())
    print("Data Size:", df.shape)
    plot_histograms(df, "Histograms Before Robust Outlier Removal")
    df_robust_removed = df[~((df[numeric_columns] < lower_bound) | 
                             (df[numeric_columns] > upper_bound)).any(axis=1)]
    print("\nAfter Robust Outlier Removal:")
    print(df_robust_removed[numeric_columns].head())
    print("Data Size:", df_robust_removed.shape)
    plot_histograms(df_robust_removed, "Histograms After Robust Outlier Removal")
    return df_robust_removed

# Apply and visualize each outlier removal method
df_iqr_removed = remove_outliers_iqr(df)
plot_correlation_heatmap(df_iqr_removed, "Correlation Heatmap After IQR Outlier Removal")

df_quantile_zscore_removed = remove_outliers_quantile_zscore(df)
plot_correlation_heatmap(df_quantile_zscore_removed, "Correlation Heatmap After Quantile Z-Score Outlier Removal")

df_quantile_std_removed = remove_outliers_quantile_std(df)
plot_correlation_heatmap(df_quantile_std_removed, "Correlation Heatmap After Quantile Standard Deviation Outlier Removal")

df_knn_removed = remove_outliers_knn(df)
plot_correlation_heatmap(df_knn_removed, "Correlation Heatmap After KNN Outlier Removal")

df_lof_removed = remove_outliers_lof(df)
plot_correlation_heatmap(df_lof_removed, "Correlation Heatmap After LOF Outlier Removal")

df_robust_removed = remove_outliers_robust(df)
plot_correlation_heatmap(df_robust_removed, "Correlation Heatmap After Robust Outlier Removal")

# Final cleaned data (select any of the cleaned datasets based on preference)
df_cleaned = df_iqr_removed.copy()  # Example: using IQR cleaned data
