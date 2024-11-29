# Importing Libraries
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

# Function Definitions
def load_and_explore_data(file_path):
    """Load and explore the dataset."""
    df = pd.read_csv(file_path)
    print("First 5 rows:\n", df.head())
    print("Last 5 rows:\n", df.tail())
    print("Dataset Info:\n")
    df.info()
    print("Summary Statistics:\n", df.describe())
    print("Missing Values:\n", df.isnull().sum())
    return df

def preprocess_numeric_columns(df, num_cols):
    """Convert specified numeric columns with commas to proper numeric types."""
    def remove_comma(x):
        return x.replace(",", "")
    
    for col in num_cols:
        df[col] = pd.to_numeric(df[col].apply(remove_comma))
    return df

def visualize_data_distributions(df, num_cols):
    """Visualize distributions of numeric columns."""
    filtered_data = df.loc[(df[num_cols] < 1000).all(axis=1)][num_cols]
    filtered_data.hist(bins=100, figsize=(18, 10))
    plt.show()
    
    sns.pairplot(df[num_cols])
    plt.show()

    df.plot.scatter(x='Number of Services', y='Average Medicare Payment Amount', figsize=(10, 6), alpha=0.5, title='Number of Services vs. Average Medicare Payment Amount')
    plt.show()

def plot_boxplots(df, num_cols, title):
    """Plot boxplots for numeric columns."""
    num_plots = len(num_cols)
    cols = 2
    rows = (num_plots // cols) + (num_plots % cols > 0)
    plt.figure(figsize=(14, 8))
    plt.suptitle(title, fontsize=16, y=1.02)
    for i, col in enumerate(num_cols):
        plt.subplot(rows, cols, i + 1)
        sns.boxplot(x=df[col])
        plt.xlabel(col)
    plt.tight_layout()
    plt.show()
    
def plot_heatmap(df, num_cols):
    """Plot heatmap for numeric columns."""
    corrplot = df[num_cols].corr()
    sns.heatmap(corrplot, annot=True, xticklabels=corrplot.columns, yticklabels=corrplot.columns)

def detect_outliers_zscore(df, num_cols, threshold=4):
    """Detect outliers using the Z-Score method."""
    z_scores = np.abs((df[num_cols] - df[num_cols].mean()) / df[num_cols].std())
    outliers = z_scores > threshold
    cleaned_df = df[~outliers.any(axis=1)]
    return cleaned_df

def detect_outliers_iqr(df, num_cols, lower_quantile=0.15, upper_quantile=0.75, multiplier=1.5):
    """Detect outliers using the IQR method."""
    Q1 = df[num_cols].quantile(lower_quantile)
    Q3 = df[num_cols].quantile(upper_quantile)
    IQR = Q3 - Q1
    outliers = (df[num_cols] < (Q1 - multiplier * IQR)) | (df[num_cols] > (Q3 + multiplier * IQR))
    cleaned_df = df[~outliers.any(axis=1)]
    return cleaned_df

def compare_outlier_removal(original_len, cleaned_dfs):
    """Compare the number of outliers removed by different methods."""
    methods = ['Z-Score', 'IQR']
    outliers_removed = [original_len - len(df) for df in cleaned_dfs]

    plt.figure(figsize=(8, 6))
    plt.bar(methods, outliers_removed, color=['blue', 'green'])
    plt.title('Number of Outliers Removed by Each Method')
    plt.xlabel('Outlier Detection Method')
    plt.ylabel('Number of Outliers Removed')
    plt.show()

# Main Workflow
if __name__ == "__main__":
    # Load the data
    data_file = 'Healthcare_Providers.csv'
    df = load_and_explore_data(data_file)

    # Define numeric columns
    numeric_columns = [
        "Number of Services", "Number of Medicare Beneficiaries",
        "Number of Distinct Medicare Beneficiary/Per Day Services",
        "Average Medicare Allowed Amount", "Average Submitted Charge Amount",
        "Average Medicare Payment Amount", "Average Medicare Standardized Amount"
    ]
    
    # Unique Values in Categorical Columns
    df[['City of the Provider', 'State Code of the Provider', 'Country Code of the Provider',
    'Entity Type of the Provider', 'Provider Type', 'Medicare Participation Indicator', 
    'Place of Service', 'HCPCS Code']].nunique()

    # Analyzing National Provider Identifier and HCPCS Code Count
    df.groupby(["National Provider Identifier"])["HCPCS Code"].nunique().sort_values(ascending=False).iloc[:10]
    df.groupby(["National Provider Identifier"])["HCPCS Code"].nunique().reset_index().groupby(["HCPCS Code"]).count()
    df.groupby(["National Provider Identifier"])["HCPCS Code"].nunique().hist(bins=30, figsize=(17, 7))

    # Provider Type Distribution
    print(df.groupby(["Provider Type"])["index"].count().sort_values(ascending=True))

    # Analyzing HCPCS Code Frequency
    df.groupby(["HCPCS Code"])["index"].count().reset_index().groupby(["index"]).count().head(10)
    df.groupby(["HCPCS Code"])["index"].count().hist(bins=1000, figsize=(17, 7))
    plt.xlim(0, 500)
    plt.ylim(0, 2000)

    # Preprocess numeric columns
    df = preprocess_numeric_columns(df, numeric_columns)
    plot_heatmap(df, numeric_columns)
    visualize_data_distributions(df, numeric_columns)

    # Detect outliers using Z-Score, IQR, and LOF
    zscore_cleaned = detect_outliers_zscore(df, numeric_columns)
    plot_boxplots(zscore_cleaned, numeric_columns, "Boxplots After Z-Score Cleaning")
    iqr_cleaned = detect_outliers_iqr(df, numeric_columns)
    plot_boxplots(iqr_cleaned, numeric_columns, "Boxplots After IQR Cleaning")
    
    # Compare outlier removal results
    compare_outlier_removal(len(df), [zscore_cleaned, iqr_cleaned])

