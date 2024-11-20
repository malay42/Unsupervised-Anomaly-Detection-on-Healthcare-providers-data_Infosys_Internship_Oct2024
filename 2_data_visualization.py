# Visualization

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_cleaned = pd.read_csv("Healthcare Providers.csv")

numeric_columns = [
    'Number of Services', 'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services',
    'Average Medicare Allowed Amount', 'Average Submitted Charge Amount',
    'Average Medicare Payment Amount', 'Average Medicare Standardized Amount'
]

# Clean numeric columns temporarily by removing commas for plotting purposes
df_cleaned_temp = df_cleaned.copy()
for col in numeric_columns:
    df_cleaned_temp[col] = df_cleaned_temp[col].astype(str).str.replace(',', '').astype(float)

categorical_columns = [
    'Gender of the Provider', 'Provider Type',
]

# Plot bar plots for each categorical column
for col in categorical_columns:
    plt.figure(figsize=(10, 6))
    df_cleaned[col].value_counts().plot(kind='bar', color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

# Box Plot for Average Medicare Payment Amount
plt.figure(figsize=(10, 6))
sns.boxplot(y=df_cleaned_temp['Average Medicare Payment Amount'])
plt.title('Box Plot of Average Medicare Payment Amount')
plt.ylabel('Average Medicare Payment Amount')
plt.show()

# Heatmap for Correlation Matrix
plt.figure(figsize=(12, 8))
corr_matrix = df_cleaned_temp[numeric_columns].dropna().corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numeric Features')
plt.show()

# Bar Plot for Average Medicare Payment Amount by Provider Type
plt.figure(figsize=(10, 6))
df_cleaned_temp.groupby('Provider Type')['Average Medicare Payment Amount'].mean().plot(kind='bar', color='orange')
plt.title('Average Medicare Payment Amount by Provider Type')
plt.xlabel('Provider Type')
plt.ylabel('Average Medicare Payment Amount')
plt.xticks(rotation=45)
plt.show()

# Pie Chart for Gender Distribution
plt.figure(figsize=(8, 8))
df_cleaned['Gender of the Provider'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'orange'])
plt.title('Gender Distribution')
plt.ylabel('')
plt.show()