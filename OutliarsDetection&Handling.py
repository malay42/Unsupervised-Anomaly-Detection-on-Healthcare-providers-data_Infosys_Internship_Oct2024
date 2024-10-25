import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# Load the dataset
df = pd.read_csv('Healthcare Providers.csv')

# Specify numeric columns to analyze
numeric_columns = ['Number of Services', 'Number of Medicare Beneficiaries',
                   'Number of Distinct Medicare Beneficiary/Per Day Services',
                   'Average Medicare Allowed Amount', 'Average Submitted Charge Amount',
                   'Average Medicare Payment Amount', 'Average Medicare Standardized Amount']

# Boxplot Before outlier detection and handling
for col in numeric_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[col])
    plt.title(f'Box Plot of {col} Before Outlier Handling')
    plt.xlabel(col)
    plt.show()

# Copy dataset for separate outlier handling
df_iqr = df.copy()  # Dataset for IQR-based outlier handling
df_zscore = df.copy()  # Dataset for Z-score-based outlier handling

# IQR Outlier Detection and Handling
def iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Apply IQR method
for col in numeric_columns:
    df_iqr = iqr(df_iqr, col)

# Count and display the percentage of outliers removed by IQR
def iqr_outlier_summary(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    total_values = len(df[column])
    percentage = (len(outliers) / total_values) * 100
    return len(outliers), percentage

iqr_outlier_summary_dict = {}
for col in numeric_columns:
    outliers, percentage = iqr_outlier_summary(df, col)
    iqr_outlier_summary_dict[col] = percentage
    print(f"IQR - {col}: Number of Outliers: {outliers}, Percentage of Outliers: {percentage:.2f}%")

# Z-score Outlier Detection and Handling
def z_score(df, column, threshold=3):
    mean = df[column].mean()
    std = df[column].std()
    df['z_score'] = (df[column] - mean) / std
    return df[np.abs(df['z_score']) <= threshold].drop(columns=['z_score'])

# Apply Z-score method
for col in numeric_columns:
    df_zscore = z_score(df_zscore, col)

# Count and display the percentage of outliers removed by Z-score
def z_score_outlier_summary(df, column, threshold=3):
    mean = df[column].mean()
    std = df[column].std()
    df['z_score'] = (df[column] - mean) / std

    outliers = df[(df['z_score'] > threshold) | (df['z_score'] < -threshold)]
    total_values = len(df[column])
    percentage = (len(outliers) / total_values) * 100
    df = df.drop(columns=['z_score'])
    return len(outliers), percentage

z_score_outlier_summary_dict = {}
for col in numeric_columns:
    outliers, percentage = z_score_outlier_summary(df, col)
    z_score_outlier_summary_dict[col] = percentage
    print(f"Z-score - {col}: Number of Outliers: {outliers}, Percentage of Outliers: {percentage:.2f}%")

# Compare and merge the better dataset
iqr_total_outliers = sum(iqr_outlier_summary_dict.values())
z_score_total_outliers = sum(z_score_outlier_summary_dict.values())

if iqr_total_outliers <= z_score_total_outliers:
    df = df_iqr
    print("Using dataset with IQR outlier handling (fewer outliers detected).")
else:
    df = df_zscore
    print("Using dataset with Z-score outlier handling (fewer outliers detected).")

# Boxplot after outlier handling
for col in numeric_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[col])
    plt.title(f'Box Plot of {col} After Outlier Handling')
    plt.xlabel(col)
    plt.show()

# Pie plot for outliers by IQR
labels = [f'{col} Outliers' for col in iqr_outlier_summary_dict.keys()]
sizes = [percentage for percentage in iqr_outlier_summary_dict.values()]
colors = plt.cm.Paired(range(len(iqr_outlier_summary_dict)))

plt.figure(figsize=(8, 8))
plt.pie(sizes, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('IQR Outlier Proportion Across All Numerical Columns')
plt.axis('equal')

legend_patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

# Pie plot for outliers by Z-score
labels = [f'{col} Outliers' for col in z_score_outlier_summary_dict.keys()]
sizes = [percentage for percentage in z_score_outlier_summary_dict.values()]
colors = plt.cm.Paired(range(len(z_score_outlier_summary_dict)))

plt.figure(figsize=(8, 8))
plt.pie(sizes, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Z-score Outlier Proportion Across All Numerical Columns')
plt.axis('equal')

legend_patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
