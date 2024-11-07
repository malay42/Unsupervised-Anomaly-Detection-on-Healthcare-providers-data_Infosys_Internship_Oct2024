#importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt
#load the cleaned dataset
df = pd.read_csv('cleaned_data.csv')
#converting numeric columns from object type to numeric type
df['Number of Services'] = pd.to_numeric(df['Number of Services'], errors='coerce')
df['Number of Medicare Beneficiaries'] = pd.to_numeric(df['Number of Medicare Beneficiaries'], errors='coerce')
df['Zip Code of the Provider'] = pd.to_numeric(df['Zip Code of the Provider'], errors='coerce')
df['Average Medicare Allowed Amount'] = pd.to_numeric(df['Average Medicare Allowed Amount'], errors='coerce')
df['Average Submitted Charge Amount'] = pd.to_numeric(df['Average Submitted Charge Amount'], errors='coerce')
df['Average Medicare Payment Amount'] = pd.to_numeric(df['Average Medicare Payment Amount'], errors='coerce')
df['Average Medicare Standardized Amount'] = pd.to_numeric(df['Average Medicare Standardized Amount'], errors='coerce')
df['Number of Distinct Medicare Beneficiary/Per Day Services']=pd.to_numeric(df['Number of Distinct Medicare Beneficiary/Per Day Services'],errors='coerce')
#printing the datatype of necessary column
print(df['Average Medicare Allowed Amount'].dtypes)
print(df['Average Submitted Charge Amount'].dtypes)
print(df['Average Medicare Payment Amount'].dtypes)
print(df['Average Medicare Standardized Amount'].dtypes)
print(df['Number of Distinct Medicare Beneficiary/Per Day Services'].dtypes)
print(df['Number of Services'].dtypes)

# Count plot for Gender of the Provider
plt.figure(figsize=(4, 2))
sns.countplot(data=df, x='Gender of the Provider')
plt.title('Count of Providers by Gender')
plt.xlabel('Gender of the Provider')
plt.ylabel('Count')
plt.show()

# Count plot for Provider Type
plt.figure(figsize=(12, 6))
sns.countplot(data=df, y='Provider Type', order=df['Provider Type'].value_counts().index)
plt.title('Count of Providers by Type')
plt.xlabel('Count')
plt.ylabel('Provider Type')
plt.show()

# All in one histogram
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
numerical_columns = numerical_columns.drop(['Zip Code of the Provider', 'index', 'National Provider Identifier'])
print(df[numerical_columns].describe())
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    print(f"Value counts for {col}:\n", df[col].value_counts(), "\n")
df[numerical_columns].hist(bins=15, figsize=(15, 10), layout=(4, 3))
plt.tight_layout()
for ax in plt.gcf().axes:
    ax.set_xlabel(ax.get_title())  # Set title as the X label
    ax.set_ylabel('Frequency')     # Set Y label as Frequency
plt.show()

#initalizing an array of all the numeric columns
numeric=['Number of Services', 'Average Medicare Allowed Amount',
        'Average Submitted Charge Amount', 
        'Average Medicare Payment Amount',
        'Average Medicare Standardized Amount', 
        'Number of Distinct Medicare Beneficiary/Per Day Services', 
        'Number of Medicare Beneficiaries']
#data distribution in numric columns
plt.figure(figsize=(15, 5))
plt.suptitle('Data Distribution in Numeric Columns', fontsize=20, fontweight='bold')

# Iterate through each numeric column and create subplots
for i, col in enumerate(numeric):
    plt.subplot(3, 3, i + 1)
    
    # Create a histogram with a kernel density estimate (KDE)
    sns.histplot(df[col], kde=True, bins=30)  # Set bins according to your preference
    
    # Draw vertical lines for mean and median
    plt.axvline(df[col].mean(), color='blue', linestyle='dashed', linewidth=2, label='Mean')
    plt.axvline(df[col].median(), color='red', linestyle='dashed', linewidth=2, label='Median')  # Corrected here
    
    # Set the labels and legend
    plt.xlabel(col)
    plt.legend()
    
# Adjust the layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust to leave space for the suptitle
plt.show()

# Scatter Plots for specific relationships
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='Average Medicare Allowed Amount', y='Number of Services', hue='Provider Type', style='Gender of the Provider')
plt.title('Scatter Plot of Average Medicare Allowed Amount vs Number of Services')
plt.xlabel('Average Medicare Allowed Amount')
plt.ylabel('Number of Services')
plt.legend()
plt.show()


# Grouped Bar Chart for Average Payment Amount by Provider Type
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x='Provider Type', y='Average Medicare Payment Amount', ci=None)
plt.title('Average Medicare Payment Amount by Provider Type')
plt.xlabel('Provider Type')
plt.ylabel('Average Medicare Payment Amount')
plt.xticks(rotation=45)
plt.show()


# Plotting histograms for the numerical columns
plt.figure(figsize=(15, 10))
plt.suptitle('Data Distribution in Numeric Columns before Outlier Removal', fontsize=20, fontweight='bold')

for i, col in enumerate(numeric):
    plt.subplot(3, 3, i + 1)
    sns.histplot(df[col], kde=True, bins=30)  # Histogram with KDE
    plt.axvline(df[col].mean(), color='blue', linestyle='dashed', linewidth=2, label='Mean')  # Mean line
    plt.axvline(df[col].median(), color='red', linestyle='dashed', linewidth=2, label='Median')  # Median line
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
  # Adjust layout to fit title
plt.show()


# Boxplots for numerical variables to check for outliers
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric, 1):
    plt.subplot(4, 3, i)
    sns.boxplot(data=df, y=col)
    plt.title(f"Boxplot of {col}")
    plt.xlabel(col)
    plt.ylabel('Value')
plt.tight_layout()
plt.show()


# HeatMap for bivariate analysis
df_numeric = df[numeric]
correlation_matrix = df_numeric.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Heatmap of Correlation Matrix for Numerical Variables")
plt.xlabel("Numerical Variables")
plt.ylabel("Numerical Variables")
plt.show()


#Save the cleaned data to a new CSV file
df.to_csv('vis_data.csv', index=False)
print("\nData visualisation complete. data saved as 'vis_data.csv'.")