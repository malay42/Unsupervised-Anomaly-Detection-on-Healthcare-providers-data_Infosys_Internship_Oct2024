# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned data
file_path = "C:/Users/shiva/Desktop/Project 1/Cleaned_Healthcare_Providers.csv"
df = pd.read_csv(file_path)

# Set seaborn theme for all plots
sns.set_theme(style="whitegrid")

# Clean and standardize the 'Gender of the Provider' column for consistent values
df['Gender of the Provider'] = (
    df['Gender of the Provider']
    .str.upper()  # Convert to uppercase for consistency
    .str.replace('.', '', regex=False)  # Remove periods
    .str.strip()  # Remove any leading/trailing whitespace
)

# Standardize "Credentials of the Provider" column
df['Credentials of the Provider'] = (
    df['Credentials of the Provider']
    .str.upper()
    .str.replace('.', '', regex=False)
    .str.strip()
    .replace(['M.D', 'MD', 'm.d.'], 'MD')  # Standardize all MD formats in credentials
)

# List of numeric columns for easy reference in plots
numeric_columns = [
    'Number of Services', 'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services',
    'Average Medicare Allowed Amount', 'Average Submitted Charge Amount',
    'Average Medicare Payment Amount', 'Average Medicare Standardized Amount'
]

# Display cleaned data information
print(df.info())
print(df.describe())

# Plot distribution of numeric columns
for column in numeric_columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()

# 1. Histogram of Numeric Columns
df[numeric_columns].hist(bins=30, figsize=(15, 10))
plt.suptitle('Histograms of Numeric Columns', fontsize=18)
plt.show()

# 2. Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df[numeric_columns].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap', fontsize=18)
plt.show()

# 3. Pie Chart: Distribution of Gender of the Provider
gender_counts = df['Gender of the Provider'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
plt.title('Distribution of Gender of the Provider', fontsize=16)
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
plt.show()

# 4. Histogram of Encoded Numerical Columns
df[numeric_columns].hist(bins=30, figsize=(15, 10))
plt.suptitle('Histograms of Encoded Numerical Columns', fontsize=18)
plt.show()

# 5. Line Plot: Number of Services vs. Average Medicare Payment Amount
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Number of Services', y='Average Medicare Payment Amount')
plt.title('Line Plot: Number of Services vs Average Medicare Payment Amount', fontsize=16)
plt.xlabel('Number of Services')
plt.ylabel('Average Medicare Payment Amount')
plt.show()

# 6. Histogram: Number of Services by Ambulance Service Provider
if 'Provider Type_Ambulance Service Provider' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Number of Services', hue='Provider Type_Ambulance Service Provider', bins=30, kde=True)
    plt.title('Histogram: Number of Services by Ambulance Service Provider', fontsize=16)
    plt.xlabel('Number of Services')
    plt.ylabel('Count')
    plt.show()

# 7. Distribution of Provider Type
plt.figure(figsize=(12, 8))
sns.countplot(data=df, y='Provider Type', order=df['Provider Type'].value_counts().index)
plt.title('Distribution of Provider Type', fontsize=16)
plt.xlabel('Count')
plt.ylabel('Provider Type')
plt.show()

# 8. Pie Chart: Top 10 Credentials of the Provider
top_credentials = df['Credentials of the Provider'].value_counts().head(10)
plt.figure(figsize=(10, 6))
plt.pie(top_credentials, labels=top_credentials.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('coolwarm', 10))
plt.title('Top 10 Credentials of the Provider', fontsize=16)
plt.axis('equal')
plt.show()

# 9. Box Plot for Numeric Columns
plt.figure(figsize=(15, 10))
df[numeric_columns].plot(kind='box', subplots=True, layout=(3, 3), figsize=(15, 10), notch=True, patch_artist=True)
plt.suptitle('Box Plot for Numeric Columns', fontsize=18)
plt.show()

# 10. Bar Plot: Top 10 Provider Types
top_provider_types = df['Provider Type'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_provider_types.values, y=top_provider_types.index, palette="viridis")
plt.title('Top 10 Provider Types', fontsize=16)
plt.xlabel('Count')
plt.ylabel('Provider Type')
plt.show()

# 11. Scatter Plot (Example: Number of Services vs Average Medicare Payment Amount)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Number of Services', y='Average Medicare Payment Amount', hue='Gender of the Provider')
plt.title('Scatter Plot: Number of Services vs Average Medicare Payment Amount', fontsize=16)
plt.xlabel('Number of Services')
plt.ylabel('Average Medicare Payment Amount')
plt.legend(title='Gender of the Provider')
plt.show()

# 12. Cluster Bar Plot by Provider Type and Gender
plt.figure(figsize=(14, 8))
sns.countplot(data=df, x='Provider Type', hue='Gender of the Provider', order=df['Provider Type'].value_counts().index)
plt.title('Cluster Bar Plot by Provider Type and Gender', fontsize=16)
plt.xticks(rotation=90)
plt.show()

# 13. Reduced Facet Grid for High-Frequency Provider Types
top_provider_types = df['Provider Type'].value_counts().nlargest(4).index
filtered_df = df[df['Provider Type'].isin(top_provider_types)]

g = sns.FacetGrid(filtered_df, col='Gender of the Provider', row='Provider Type', margin_titles=True, height=3, aspect=1.5)
g.map(sns.histplot, 'Average Medicare Payment Amount', bins=20, kde=True)
g.set_axis_labels('Average Medicare Payment Amount', 'Frequency')
g.set_titles(col_template="{col_name}", row_template="{row_name}")
g.fig.suptitle("Distribution of Average Medicare Payment Amount by Gender and Provider Type (Top 4)", fontsize=16)
plt.tight_layout()
plt.show()
