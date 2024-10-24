# %%
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Healthcare Providers.csv')

# Distribution of Medicare Payment Amounts
plt.figure(figsize=(8, 6))
plt.hist(df['Average Medicare Payment Amount'], bins=1000, color='skyblue', edgecolor='green', log=True)
plt.title('Distribution of Medicare Payment Amounts (Log Scale)')
plt.xlabel('Medicare Payment Amount (Log Scale)')
plt.ylabel('Log of Frequency')
plt.grid(True)
plt.show()



# %%
# Total Number of Services by Provider Type
plt.figure(figsize=(10, 6))
top_providers = df.groupby('Provider Type')['Number of Services'].sum().nlargest(40)
top_providers.plot(kind='bar', color='lightgreen')
plt.title('Provider Types by Number of Services')
plt.xlabel('Provider Type')
plt.ylabel('Total Number of Services')
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.show()


# %%
# Average Medicare Payment by Gender
plt.figure(figsize=(8, 6))
avg_payment_by_gender = df.groupby('Gender of the Provider')['Average Medicare Payment Amount'].mean()
avg_payment_by_gender.plot(kind='bar', color=['pink', 'lightblue'])
plt.title('Average Medicare Payment by Gender')
plt.xlabel('Gender')
plt.ylabel('Average Medicare Payment Amount')
plt.xticks(rotation=0)
plt.grid(True)
plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

# Load the dataset
df = pd.read_csv('Healthcare Providers.csv')

# Convert relevant columns to numeric for correlation analysis
numeric_columns = [
    'Number of Services', 'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services',
    'Average Medicare Allowed Amount', 'Average Submitted Charge Amount',
    'Average Medicare Payment Amount', 'Average Medicare Standardized Amount'
]

# Convert columns to numeric, coercing errors to handle non-numeric values
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Generate a correlation matrix for the numeric columns
correlation_matrix = df[numeric_columns].corr()

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()


# %%
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='Average Submitted Charge Amount', 
    y='Average Medicare Payment Amount', 
    data=df, 
    alpha=0.6
)
plt.title('Average Submitted Charge Amount vs. Average Medicare Payment Amount')
plt.xlabel('Average Submitted Charge Amount')
plt.ylabel('Average Medicare Payment Amount')
plt.show()

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('Healthcare Providers.csv')

# Assuming you have columns for 'State' and 'Average Medicare Allowed Amount'
# Modify the column names if needed based on your CSV file structure
plt.figure(figsize=(14, 8))
sns.boxplot(x='State Code of the Provider', y='Average Medicare Allowed Amount', data=data)

# Add labels and title
plt.xticks(rotation=90)
plt.title('Distribution of Average Medicare Allowed Amount by State')
plt.xlabel('State')
plt.ylabel('Average Medicare Allowed Amount')

# Save and display the plot
plt.tight_layout()
plt.show()



