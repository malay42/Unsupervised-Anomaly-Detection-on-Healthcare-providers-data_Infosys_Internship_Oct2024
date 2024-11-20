# Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# Load the dataset
def load_data():
    try:
        df = pd.read_csv('Cleaned_Healthcare_Providers.csv')
        return df
    except FileNotFoundError:
        print("The specified file was not found.")
        exit()

# Convert numeric columns (specify these if needed)
numeric_columns = ['Number of Services', 'Number of Medicare Beneficiaries',
                   'Number of Distinct Medicare Beneficiary/Per Day Services',
                   'Average Medicare Allowed Amount', 'Average Submitted Charge Amount',
                   'Average Medicare Payment Amount', 'Average Medicare Standardized Amount']

# Convert string columns with commas to numeric
def convert_numeric_columns(df):
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')
    return df

# Plot boxplots for the specified numeric columns.
def plot_boxplots(df, numeric_columns, title_suffix='Before'):
    for col in numeric_columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df[col])
        plt.title(f'Box Plot of {col} {title_suffix} Outlier Handling')
        plt.xlabel(col)
        plt.show()

# Detect outliers using IQR method.
def iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Detect outliers using Z-score method.
def z_score(df, column, threshold=3):
    mean = df[column].mean()
    std = df[column].std()
    return df[(df[column] - mean).abs() <= threshold * std]

# Summarize the percentage of outliers removed by IQR.
def summarize_outliers_iqr(df, numeric_columns):
    summary = {}
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        total_values = len(df[col])
        percentage = (len(outliers) / total_values) * 100
        summary[col] = percentage
    return summary

# Summarize the percentage of outliers removed by Z-score.
def summarize_outliers_zscore(df, numeric_columns):
    summary = {}
    for col in numeric_columns:
        mean = df[col].mean()
        std = df[col].std()
        outliers = df[(df[col] - mean).abs() > 3 * std]
        total_values = len(df[col])
        percentage = (len(outliers) / total_values) * 100
        summary[col] = percentage
    return summary

# Compare datasets from IQR and Z-score methods and return the one with fewer outliers.
def compare_datasets(iqr_summary, z_score_summary):
    iqr_total_outliers = sum(iqr_summary.values())
    z_score_total_outliers = sum(z_score_summary.values())
    return iqr_total_outliers <= z_score_total_outliers

# Main function to perform outlier detection and handling.
def main():
    # Load and clean the data
    df = load_data()
    df = convert_numeric_columns(df)

    # Plot boxplots before handling outliers
    plot_boxplots(df, numeric_columns, title_suffix='Before')

    # Copy dataset for separate outlier handling
    df_iqr = df.copy()  # Dataset for IQR-based outlier handling
    df_zscore = df.copy()  # Dataset for Z-score-based outlier handling

    # Apply IQR method
    for col in numeric_columns:
        df_iqr = iqr(df_iqr, col)

    # Summarize IQR outliers
    iqr_summary = summarize_outliers_iqr(df, numeric_columns)
    for col, percentage in iqr_summary.items():
        print(f"IQR - {col}: Percentage of Outliers: {percentage:.2f}%")

    # Apply Z-score method
    for col in numeric_columns:
        df_zscore = z_score(df_zscore, col)

    # Summarize Z-score outliers
    z_score_summary = summarize_outliers_zscore(df, numeric_columns)
    for col, percentage in z_score_summary.items():
        print(f"Z-score - {col}: Percentage of Outliers: {percentage:.2f}%")

    # Compare datasets and select the better one
    if compare_datasets(iqr_summary, z_score_summary):
        df = df_iqr
        print("Using dataset with IQR outlier handling (fewer outliers detected).")
    else:
        df = df_zscore
        print("Using dataset with Z-score outlier handling (fewer outliers detected).")

    # Plot boxplots after handling outliers
    plot_boxplots(df, numeric_columns, title_suffix='After')

    # Pie plot for outlier proportions
    labels = [f'{col} Outliers' for col in iqr_summary.keys()]
    sizes = list(iqr_summary.values())
    colors = plt.cm.Paired(range(len(labels)))

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('IQR Outlier Proportion Across All Numerical Columns')
    plt.axis('equal')

    legend_patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
    plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    # Optionally save the cleaned dataset
    df.to_csv('Cleaned_Healthcare_Providers_Handled_Outliers.csv', index=False)

if __name__ == "__main__":
    main()
