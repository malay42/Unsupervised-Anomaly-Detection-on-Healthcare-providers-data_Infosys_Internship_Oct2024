# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('iqr_cleaned_file.csv')

# Clean up gender data - make it consistent
df['Gender of the Provider'] = df['Gender of the Provider'].str.upper().str.strip()

# List of columns containing numerical data for analysis
number_cols = [
    'Number of Services',
    'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services',
    'Average Medicare Allowed Amount',
    'Average Submitted Charge Amount',
    'Average Medicare Payment Amount',
    'Average Medicare Standardized Amount'
]

# Create and display all visualizations
# 1. Distribution of numeric columns
for column in number_cols:
    plt.figure(figsize=(10, 5))
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()

# 2. Gender vs Average Submitted Charge
plt.figure(figsize=(10, 8))
sns.barplot(x='Gender of the Provider', y='Average Submitted Charge Amount', data=df, estimator='mean')
plt.title('Average Submitted Charge Amount by Gender of the Provider')
plt.xlabel('Gender of the Provider')
plt.ylabel('Average Submitted Charge Amount')
plt.show()

# 3. Entity Type vs Number of Services
plt.figure(figsize=(10, 6))
sns.barplot(x='Entity Type of the Provider', y='Number of Services', data=df, estimator='sum')
plt.title('Total Number of Services by Entity Type of Provider')
plt.xlabel('Entity Type of Provider')
plt.ylabel('Total Number of Services')
plt.show()

# 4. HCPCS Drug Indicator vs Services
plt.figure(figsize=(8, 6))
sns.barplot(x='HCPCS Drug Indicator', y='Number of Services', data=df, estimator='sum')
plt.title('Total Number of Services by HCPCS Drug Indicator')
plt.xlabel('HCPCS Drug Indicator')
plt.ylabel('Total Number of Services')
plt.show()

# 5. Correlation Heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = df[number_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# 6. Scatter Plot: Charge vs Payment by Gender
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Average Submitted Charge Amount', 
                y='Average Medicare Payment Amount', hue='Gender of the Provider')
plt.title('Average Submitted Charge vs. Average Medicare Payment')
plt.show()

# 7. Top 10 HCPCS Codes
plt.figure(figsize=(10, 6))
top_hcpcs = df['HCPCS Code'].value_counts().nlargest(10)
sns.barplot(x=top_hcpcs.index, y=top_hcpcs.values)
plt.title('Top 10 HCPCS Codes')
plt.ylabel('Frequency')
plt.xlabel('HCPCS Code')
plt.xticks(rotation=45)
plt.show()

# 8. Provider Type Distribution
plt.figure(figsize=(12, 8))
sns.countplot(data=df, y='Provider Type', order=df['Provider Type'].value_counts().index)
plt.title('Distribution of Provider Type')
plt.xlabel('Count')
plt.ylabel('Provider Type')
plt.show()

# 9. State Distribution
plt.figure(figsize=(15, 6))
state_counts = df['State Code of the Provider'].value_counts()
sns.barplot(x=state_counts.index, y=state_counts.values)
plt.title('Distribution of Providers by State')
plt.xlabel('State Code')
plt.ylabel('Number of Providers')
plt.xticks(rotation=45)
plt.show()

# 10. Box Plots for Numeric Columns
plt.figure(figsize=(15, 10))
df[number_cols].plot(kind='box', subplots=True, layout=(3, 3), 
                    figsize=(15, 10), notch=True, patch_artist=True)
plt.suptitle('Box Plot for Numeric Columns')
plt.tight_layout()
plt.show()

# 11. Medicare Participation Distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Medicare Participation Indicator')
plt.title('Distribution of Medicare Participation')
plt.xlabel('Medicare Participation Indicator')
plt.ylabel('Count')
plt.show()

# 12. Pair Plot for Key Metrics
sns.pairplot(df[[
    'Number of Services',
    'Number of Medicare Beneficiaries',
    'Average Medicare Allowed Amount',
    'Average Submitted Charge Amount',
    'Average Medicare Payment Amount'
]])
plt.show()
