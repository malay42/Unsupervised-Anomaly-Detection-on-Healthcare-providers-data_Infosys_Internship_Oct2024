from sklearn.preprocessing import LabelEncoder
import pandas as pd

label_encoding_columns = [
    'Gender of the Provider',
    'Entity Type of the Provider',
    'Medicare Participation Indicator',
    'HCPCS Drug Indicator',
    'Credentials of the Provider',  
    'HCPCS Code',  
    'HCPCS Description'  
]

one_hot_encoding_columns = [
    'Provider Type',
    'Place of Service',
    'State Code of the Provider',
    'Country Code of the Provider'
]


label_encoder = LabelEncoder()


for col in label_encoding_columns:
    if col in data_dropped.columns:  
        data_dropped[col] = label_encoder.fit_transform(data_dropped[col])


data_dropped = pd.get_dummies(data_dropped, columns=one_hot_encoding_columns, drop_first=False)


data_dropped = data_dropped.replace({True: 'T', False: 'F'})


print(data_dropped.head())
print(data_dropped.isnull().sum())

numerical_columns_count = data_dropped.select_dtypes(include=['int64', 'float64']).shape[1]


print("Number of numerical columns:", numerical_columns_count)
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(style="whitegrid")


numerical_columns = [
    'Average Medicare Payment Amount',
    'Average Medicare Allowed Amount',
    'Average Submitted Charge Amount',
    'Number of Services',
    'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services',
    'Average Medicare Standardized Amount'
]

plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(3, 3, i)
    plt.hist(data_dropped[col], bins=30, edgecolor='black')
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(True)

plt.tight_layout()
plt.show()


categorical_columns = [
    'Gender of the Provider',
    'Entity Type of the Provider',
    'Medicare Participation Indicator',
    'HCPCS Drug Indicator'
]

plt.figure(figsize=(15, 10))
for i, col in enumerate(categorical_columns, 1):
    plt.subplot(2, 2, i)
    sns.countplot(data=data_dropped, x=col)
    plt.title(f'Count of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
plt.scatter(data_dropped['Number of Medicare Beneficiaries'], data_dropped['Average Medicare Payment Amount'], alpha=0.5)
plt.title('Scatter plot of Number of Medicare Beneficiaries vs Average Medicare Payment Amount')
plt.xlabel('Number of Medicare Beneficiaries')
plt.ylabel('Average Medicare Payment Amount')
plt.grid(True)
plt.show()



plt.figure(figsize=(10, 6))
sns.heatmap(data_dropped[numerical_columns].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Variables')
plt.savefig('vis9.png') 
# Save the figure as a PNG file
# Display only numerical columns
numerical_columns = data_dropped.select_dtypes(include=['number'])

# Show the first few rows of the numerical columns
numerical_columns.head()