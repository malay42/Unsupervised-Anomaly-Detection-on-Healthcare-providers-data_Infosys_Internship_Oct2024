import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("C:\\Users\\malla\\Downloads\\archive\\Healthcare Providers.csv")

# Convert numeric columns to float
numeric_columns = [
'Number of Services', 'Number of Medicare Beneficiaries',
'Number of Distinct Medicare Beneficiary/Per Day Services',
'Average Medicare Allowed Amount', 'Average Submitted Charge Amount',
'Average Medicare Payment Amount', 'Average Medicare Standardized Amount'
]

# Convert columns to numeric
for col in numeric_columns:
df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')

# Define the overall color palette
sns.set_palette("Set2")

# Bivariate Analysis

# Heatmap with enhanced color and layout
plt.figure(figsize=(12, 10))
correlation_matrix = df[numeric_columns].corr()
sns.heatmap(
correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f',
cbar_kws={'shrink': .8}, square=True, linewidths=.5
)
plt.title('Correlation Heatmap of Numeric Columns', fontsize=16)
plt.xticks(rotation=45)
plt.show()

# Boxplot of Medicare Payment by Gender
plt.figure(figsize=(10, 8))
sns.boxplot(data=df, x='Gender of the Provider', y='Average Medicare Payment Amount', palette="coolwarm")
plt.title('Boxplot of Medicare Payment by Gender', fontsize=16)
plt.xlabel('Gender of the Provider')
plt.ylabel('Average Medicare Payment Amount')
plt.show()

# Scatter plot of Average Medicare Payment Amount vs. Number of Services with trend line
plt.figure(figsize=(10, 6))
sns.regplot(
data=df, x='Average Medicare Payment Amount', y='Number of Services',
scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'}
)
plt.title('Average Medicare Payment Amount vs. Number of Services', fontsize=16)
plt.xlabel('Average Medicare Payment Amount')
plt.ylabel('Number of Services')
plt.show()


# Univariate Analysis


# Pair Plot with KDE on the diagonal
sns.pairplot(df[numeric_columns], diag_kind='kde', plot_kws={'alpha': 0.6})
plt.suptitle('Pair Plot of Numeric Features', y=1.02, fontsize=16)
plt.show()

# Enhanced Histograms for each numeric column
for col in numeric_columns:
plt.figure(figsize=(10, 6))
sns.histplot(df[col], kde=True, bins=30, color='skyblue')
plt.title(f'Distribution of {col}', fontsize=14)
plt.xlabel(col)
plt.ylabel('Frequency')
plt.show()

# Enhanced Boxplots for each numeric column
plt.figure(figsize=(14, 10))
for i, col in enumerate(numeric_columns, 1):
plt.subplot(3, 3, i)
sns.boxplot(data=df, x=col, color="lightblue")
plt.title(f'Boxplot of {col}', fontsize=12)
plt.tight_layout()
plt.show()


