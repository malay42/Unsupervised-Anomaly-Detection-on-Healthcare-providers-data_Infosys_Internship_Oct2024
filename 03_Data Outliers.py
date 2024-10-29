import matplotlib.pyplot as plt
import pandas as pd

# Load your dataset
df = pd.read_csv('Healthcare Providers.csv')

# Clean numerical columns by removing commas and converting to numbers
df['Average Medicare Allowed Amount'] = df['Average Medicare Allowed Amount'].replace({',': ''}, regex=True)
df['Average Medicare Allowed Amount'] = pd.to_numeric(df['Average Medicare Allowed Amount'], errors='coerce')

# Create the box plot to identify outliers
plt.figure(figsize=(10, 6))
plt.boxplot(df['Average Medicare Allowed Amount'], vert=False, patch_artist=True)
plt.title('Box Plot of Average Medicare Allowed Amount', fontsize=16)
plt.xlabel('Average Medicare Allowed Amount', fontsize=14)
plt.grid(True)

# Show the plot
plt.show()


cols_to_clean = ['Number of Services', 'Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 
                 'Average Medicare Payment Amount', 'Average Medicare Standardized Amount', 'Number of Medicare Beneficiaries']
for col in cols_to_clean:
    df[col] = df[col].replace({',': ''}, regex=True)
    df[col] = pd.to_numeric(df[col], errors='coerce')

# List of columns to plot
columns_to_plot = ['Number of Services', 'Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 
                   'Average Medicare Payment Amount', 'Average Medicare Standardized Amount', 'Number of Medicare Beneficiaries']

# Create subplots for the box plots
plt.figure(figsize=(15, 10))
for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(2, 3, i)
    plt.boxplot(df[column], vert=False, patch_artist=True)
    plt.title(f'Box Plot of {column}', fontsize=12)
    plt.grid(True)

plt.tight_layout()
plt.show()


for col in cols_to_clean:
    df[col] = df[col].replace({',': ''}, regex=True)
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Plot box plots for all numerical columns
plt.figure(figsize=(15, 10))
df[cols_to_clean].boxplot(patch_artist=True, vert=False)
plt.title('Box Plots for Numerical Columns (Detecting Outliers)', fontsize=16)
plt.xlabel('Values', fontsize=14)
plt.grid(True)
plt.show()

# Function to detect outliers using IQR
def detect_outliers(column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

# Detect outliers for each numerical column and print them
for col in cols_to_clean:
    outliers = detect_outliers(col)
    print(f"Outliers in '{col}':")
    print(outliers[[col]], '\n')


# Box plot for 'Average Medicare Payment Amount'
plt.figure(figsize=(10, 6))
plt.boxplot(df['Average Medicare Payment Amount'], vert=False, patch_artist=True)
plt.title('Box Plot of Average Medicare Payment Amount (Detecting Outliers)', fontsize=16)
plt.xlabel('Average Medicare Payment Amount', fontsize=14)
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt

# Function to detect outliers and plot boxplot for outliers
def detect_and_plot_outliers(column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    # Plotting Box Plot
    plt.figure(figsize=(10, 6))
    plt.boxplot(df[column], vert=False, patch_artist=True)
    plt.title(f'Box Plot of {column} (Detecting Outliers)', fontsize=16)
    plt.xlabel(column, fontsize=14)
    plt.grid(True)
    plt.show()
    
    return outliers

# Detect and visualize outliers for each numerical column
for col in cols_to_clean:
#     print(f"Outliers in '{col}':")
    outliers = detect_and_plot_outliers(col)
#     print(outliers[[col]], '\n')

# -----------------------------------------------------------



# Function to plot histogram of values with outliers marked
def plot_histogram_outliers(column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Detect outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

    # Plotting Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df[column], bins=30, color='lightblue', edgecolor='black')
    
    # Overlay outliers in red
    plt.hist(outliers[column], bins=30, color='red', edgecolor='black', alpha=0.6)
    
    plt.title(f'Histogram of {column} with Outliers Highlighted', fontsize=16)
    plt.xlabel(column, fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True)
    plt.show()

# Plot histograms for each column
for col in cols_to_clean:
    plot_histogram_outliers(col)


import pandas as pd
import numpy as np

def detect_outliers_zscore(df, threshold=3):
    """
    Detect outliers using the Z-score method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    threshold (float): The Z-score value above which data points are considered outliers.
    
    Returns:
    pd.DataFrame: DataFrame with an additional column indicating whether the row is an outlier.
    """
    # Calculate Z-scores for numerical columns
    df_zscore = df.select_dtypes(include=[np.number])
    z_scores = np.abs((df_zscore - df_zscore.mean()) / df_zscore.std())
    
    # Create a new column 'Outlier' to flag rows where any Z-score exceeds the threshold
    df['Outlier'] = (z_scores > threshold).any(axis=1)
    
    return df

# Example usage:
# Assuming df is your DataFrame
df_outliers = detect_outliers_zscore(df, threshold=3)

# To see outliers only:
outliers = df_outliers[df_outliers['Outlier'] == True]
print(outliers)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_boxplots(df):
    """
    Plot boxplots for each numerical column in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    """
    # Select numerical columns
    num_columns = df.select_dtypes(include=[np.number]).columns
    
    # Set up the plot
    plt.figure(figsize=(15, 10))
    
    # Plot a boxplot for each numerical column
    for i, col in enumerate(num_columns, 1):
        plt.subplot(len(num_columns)//3 + 1, 3, i)  # Create a subplot
        sns.boxplot(data=df, y=col)
        plt.title(f'Boxplot of {col}')
    
    plt.tight_layout()  # Adjust layout
    plt.show()

# Example usage:
plot_boxplots(df)


import matplotlib.pyplot as plt
import numpy as np

# Function to detect outliers using Z-score and plot boxplot for outliers
def detect_and_plot_outliers_zscore(column, threshold=3):
    mean_col = np.mean(df[column])
    std_col = np.std(df[column])
    
    # Calculate Z-scores for the column
    z_scores = (df[column] - mean_col) / std_col
    
    # Detect outliers based on the Z-score threshold
    outliers = df[np.abs(z_scores) > threshold]
    
    # Plotting Box Plot
    plt.figure(figsize=(10, 6))
    plt.boxplot(df[column], vert=False, patch_artist=True)
    plt.title(f'Box Plot of {column} (Detecting Outliers using Z-Score)', fontsize=16)
    plt.xlabel(column, fontsize=14)
    plt.grid(True)
    plt.show()
    
    return outliers

# Detect and visualize outliers for each numerical column
for col in cols_to_clean:
#     print(f"Outliers in '{col}':")
    outliers = detect_and_plot_outliers_zscore(col, threshold=3)
#     print(outliers[[col]], '\n')


import matplotlib.pyplot as plt
import numpy as np

# Function to detect and remove outliers using Z-score
def remove_outliers_zscore(df, column, threshold=3):
    mean_col = np.mean(df[column])
    std_col = np.std(df[column])
    
    # Calculate Z-scores for the column
    z_scores = (df[column] - mean_col) / std_col
    
    # Remove outliers based on Z-score threshold
    df_clean = df[np.abs(z_scores) <= threshold]
    
    return df_clean

# Function to plot before and after outlier removal
def plot_before_after_outliers(df, column, threshold=3):
    # 1. Plot before removing outliers
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)  # First plot for "Before"
    plt.boxplot(df[column], vert=False, patch_artist=True)
    plt.title(f'Box Plot of {column} (Before Outlier Removal)', fontsize=16)
    plt.xlabel(column, fontsize=14)
    plt.grid(True)
    
    # 2. Remove outliers using Z-score method
    df_clean = remove_outliers_zscore(df, column, threshold)
    
    # 3. Plot after removing outliers
    plt.subplot(1, 2, 2)  # Second plot for "After"
    plt.boxplot(df_clean[column], vert=False, patch_artist=True)
    plt.title(f'Box Plot of {column} (After Outlier Removal)', fontsize=16)
    plt.xlabel(column, fontsize=14)
    plt.grid(True)
    
    # Show both plots
    plt.tight_layout()
    plt.show()
    
    return df_clean

# Example usage:
# Assuming 'df' is your DataFrame and 'column_name' is the column to be analyzed
for col in cols_to_clean:
    df_cleaned = plot_before_after_outliers(df, col, threshold=3)


import matplotlib.pyplot as plt
import numpy as np

# Function to detect and remove outliers using IQR method
def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column to remove outliers from.
    
    Returns:
    pd.DataFrame: A DataFrame without the outliers in the specified column.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Remove outliers
    df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return df_clean

# Function to plot before and after outlier removal using IQR method
def plot_before_after_outliers_iqr(df, column):
    """
    Plot box plots before and after outlier removal using the IQR method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column to detect and remove outliers from.
    """
    # 1. Plot before removing outliers
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)  # First plot for "Before"
    plt.boxplot(df[column], vert=False, patch_artist=True)
    plt.title(f'Box Plot of {column} (Before Outlier Removal)', fontsize=16)
    plt.xlabel(column, fontsize=14)
    plt.grid(True)
    
    # 2. Remove outliers using IQR method
    df_clean = remove_outliers_iqr(df, column)
    
    # 3. Plot after removing outliers
    plt.subplot(1, 2, 2)  # Second plot for "After"
    plt.boxplot(df_clean[column], vert=False, patch_artist=True)
    plt.title(f'Box Plot of {column} (After Outlier Removal)', fontsize=16)
    plt.xlabel(column, fontsize=14)
    plt.grid(True)
    
    # Show both plots
    plt.tight_layout()
    plt.show()
    
    return df_clean

# Example usage:
# Assuming 'df' is your DataFrame and 'cols_to_clean' is the list of numerical columns
for col in cols_to_clean:
    df_cleaned = plot_before_after_outliers_iqr(df, col)