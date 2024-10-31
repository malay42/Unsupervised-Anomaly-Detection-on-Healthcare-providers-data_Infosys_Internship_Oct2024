import matplotlib
matplotlib.use('Agg')  

import matplotlib.pyplot as plt
import seaborn as sns



plt.figure(figsize=(15, 12))


columns = [
    'Number of Services',
    'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services',
    'Average Medicare Allowed Amount',
    'Average Submitted Charge Amount',
    'Average Medicare Payment Amount',
    'Average Medicare Standardized Amount'
]


for i, col in enumerate(columns, 1):
    plt.subplot(3, 3, i)  # 3 rows, 3 columns
    sns.histplot(data_dropped[col], kde=True)  # Using histplot for better visual representation with kde curve
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

# Adjust layout
plt.tight_layout()

plt.savefig('output_figure.png')  

summary_stats = data_dropped.describe()
print("Summary Statistics:\n", summary_stats)


plt.figure(figsize=(18, 12))


plt.subplot(3, 3, 1)
sns.histplot(data=data_dropped, x='Number of Services', kde=False, bins=30)
plt.title('Distribution of Number of Services')


plt.subplot(3, 3, 2)
sns.histplot(data=data_dropped, x='Number of Medicare Beneficiaries', kde=False, bins=30)
plt.title('Distribution of Number of Medicare Beneficiaries')


plt.subplot(3, 3, 3)
sns.histplot(data=data_dropped, x='Number of Distinct Medicare Beneficiary/Per Day Services', kde=False, bins=30)
plt.title('Distribution of Distinct Medicare Beneficiary/Per Day Services')


plt.subplot(3, 3, 4)
sns.histplot(data=data_dropped, x='Average Medicare Allowed Amount', kde=False, bins=30)
plt.title('Distribution of Average Medicare Allowed Amount')


plt.subplot(3, 3, 5)
sns.histplot(data=data_dropped, x='Average Submitted Charge Amount', kde=False, bins=30)
plt.title('Distribution of Average Submitted Charge Amount')


plt.subplot(3, 3, 6)
sns.histplot(data=data_dropped, x='Average Medicare Payment Amount', kde=False, bins=30)
plt.title('Distribution of Average Medicare Payment Amount')


plt.subplot(3, 3, 7)
sns.histplot(data=data_dropped, x='Average Medicare Standardized Amount', kde=False, bins=30)
plt.title('Distribution of Average Medicare Standardized Amount')

plt.tight_layout()
plt.savefig('Visualisation1.png')  

plt.show()


plt.figure(figsize=(18, 12))


plt.subplot(3, 3, 1)
sns.boxplot(x=data_dropped['Number of Services'])
plt.title('Boxplot - Number of Services')


plt.subplot(3, 3, 2)
sns.boxplot(x=data_dropped['Number of Medicare Beneficiaries'])
plt.title('Boxplot - Number of Medicare Beneficiaries')


plt.subplot(3, 3, 3)
sns.boxplot(x=data_dropped['Number of Distinct Medicare Beneficiary/Per Day Services'])
plt.title('Boxplot - Distinct Medicare Beneficiary/Per Day Services')

plt.subplot(3, 3, 4)
sns.boxplot(x=data_dropped['Average Medicare Allowed Amount'])
plt.title('Boxplot - Average Medicare Allowed Amount')


plt.subplot(3, 3, 5)
sns.boxplot(x=data_dropped['Average Submitted Charge Amount'])
plt.title('Boxplot - Average Submitted Charge Amount')


plt.subplot(3, 3, 6)
sns.boxplot(x=data_dropped['Average Medicare Payment Amount'])
plt.title('Boxplot - Average Medicare Payment Amount')


plt.subplot(3, 3, 7)
sns.boxplot(x=data_dropped['Average Medicare Standardized Amount'])
plt.title('Boxplot - Average Medicare Standardized Amount')

plt.tight_layout()
plt.savefig('Visualisation2.png')  
columns_of_interest = [
    'Number of Services',
    'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services',
    'Average Medicare Allowed Amount',
    'Average Submitted Charge Amount',
    'Average Medicare Payment Amount',
    'Average Medicare Standardized Amount'
]


correlation_matrix = data_dropped[columns_of_interest].corr()


plt.figure(figsize=(10, 6))


sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})


plt.title('Correlation Heatmap of Selected Features')


plt.savefig('Visualisation3.png')  
plt.figure(figsize=(10, 6))
plt.scatter(data_dropped['Number of Medicare Beneficiaries'], data_dropped['Average Medicare Payment Amount'], alpha=0.5)
plt.title('Scatter plot of Number of Medicare Beneficiaries vs Average Medicare Payment Amount')
plt.xlabel('Number of Medicare Beneficiaries')
plt.ylabel('Average Medicare Payment Amount')
plt.grid(True)
plt.savefig('vis4.png')  
import pandas as pd
import matplotlib.pyplot as plt

categorical_columns = [
    'Gender of the Provider',
    'Entity Type of the Provider',
    'Medicare Participation Indicator',
    'HCPCS Drug Indicator'
]


for col in categorical_columns:
    
    category_counts = data_dropped[col].value_counts()

    # Create the pie chart
    plt.figure(figsize=(7, 7))
    plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
    plt.title(f'Distribution of {col}')
    plt.show()
plt.savefig('vis5.png')  
