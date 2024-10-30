# Exaploratory Data Analysis

import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy.stats import skew
from scipy.stats import kurtosis

df = pd.read_csv('Healthcare_Providers.csv')        # Loading dataset

# Univariate Analysis - Numerical Columns 
numeric_cols = ['Number of Services' ,
                'Number of Medicare Beneficiaries' ,
                'Number of Distinct Medicare Beneficiary/Per Day Services' ,
                'Average Medicare Allowed Amount' ,
                'Average Submitted Charge Amount' ,
                'Average Medicare Payment Amount', 
                'Average Medicare Standardized Amount']
                
# Measure of Central Tendency: Mean, Median, Mode
central_tendency = {}        # Create an empty dictionary to store the results
for col in numeric_cols:
    mean_value = df[col].mean()
    median_value = df[col].median()
    mode_value = df[col].mode()[0]
    central_tendency[col] = {'Mean': mean_value, 'Median': median_value, 'Mode': mode_value}
central_tendency_df = pd.DataFrame(central_tendency).T        # Convert results to a DataFrame
print(central_tendency_df)

# Measures of dispersion - Assess the spread or variability of data around the central tendency
# Range, Variance, Standard deviation, Interquartile range (IQR) 
measure_dispersion = {}        # Create an empty dictionary to store the results
for col in numeric_cols:
    range_value = df[col].max() - df[col].min()
    variance_value = df[col].var()
    std_dev_value = df[col].std()
    IQR = df[col].quantile(0.75) - df[col].quantile(0.25)
    measure_dispersion[col] = {'Range': range_value, 'Variance': variance_value, 'Standard Deviation': std_dev_value, 'IQR': IQR}
measure_dispersion_df = pd.DataFrame(measure_dispersion).T        # Convert results to a DataFrame
print(measure_dispersion_df)

# Shape of distribution: Skewness, Kurtosis
distribution = {}        # Create an empty dictionary to store the results
for col in numeric_cols:
    col_skewness = skew(df[col].dropna())               # Describe the symmetry of the data distribution
    col_kurtosis = kurtosis(df[col].dropna())           # Describe the peakedness of the data distribution
    distribution[col] = {'Skewness': col_skewness, 'Kurtosis': col_kurtosis}
distribution_df = pd.DataFrame(distribution).T        # Convert results to a DataFrame
print(distribution_df)

# Histogram: SVisualizes the frequency distribution of continuous data
df[numeric_cols].hist(bins=20, figsize=(15, 10))
plt.suptitle("Histograms of Numerical Features")
plt.show() 

# Box plot: Displays data distribution and highlights outliers through quartiles
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
plt.tight_layout()
plt.show()

# Density plot: Shows the probability distribution of a continuous variable (KDE Plot)
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 3, i)
    sns.kdeplot(df[col])
    plt.title(f"KDE Plot of {col}")
plt.tight_layout()
plt.show()

# Univariate Analysis - Categorical Columns 
categorical_cols = df.select_dtypes(include=["object"]).columns
print("Total number categorical of features: ", categorical_cols.shape[0])
print("Categorical features name: ", categorical_cols.tolist())

# Calculating Frequency Count and Mode of categorical Features 
for col in categorical_cols:
    count_value = df[col].value_counts()
    print(count_value)
    mode_value = df[col].mode()[0]
    print(f"Mode of {col} is {mode_value}\n")

categorical_cols = ['Credentials of the Provider', 'Gender of the Provider',
       'Entity Type of the Provider', 'City of the Provider',
       'State Code of the Provider', 'Provider Type',
       'Place of Service', 'HCPCS Drug Indicator']

# Bar plot: To show frequency counts of categories
def bar_chart(df, col):
    counts = df[col].value_counts()             # Get frequency counts for the categorical column
    plt.figure(figsize=(4,3))                   # Plot Bar Chart
    counts.plot(kind='bar', color='skyblue')
    plt.title(f'Bar Plot: {col}')
    plt.xlabel(f'{col}')
    plt.ylabel('Count')
    plt.show()
for col in categorical_cols:
    bar_chart(df,col)

# Pie Plot - Represents proportions of categories within a whole(pie)
def plot_pie_chart(df, col):
    counts = df[col].value_counts()   
    plt.figure(figsize=(4,4))               # Plot Pie Chart
    counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, explode=[0.05] * len(counts), colors=plt.cm.Paired.colors)
    plt.title(f'Pie Chart: {col}')
    plt.ylabel('')  # Remove y-label for better presentation
    plt.show()
for col in categorical_cols:
    plot_pie_chart(df,col)

# Bivariate Analysis - Numerical vs Numerical Data
numerical_features = df.select_dtypes(include=['float64', 'int64'])

# Pearson Correlation Coefficient - Measures the linear relationship between two continuous variables
pearson_corr = numerical_features.corr(method='pearson')
print("Pearson Correlation Coefficient:\n", pearson_corr)

# Spearman Correlation Coefficient - Assesses the rank-order relationship between variables
spearman_corr = numerical_features.corr(method='spearman')
print("\nSpearman Correlation Coefficient:\n", spearman_corr)

# Kendall Correlation Coefficient - Evaluates ordinal associations between variables
kendall_corr = numerical_features.corr(method='kendall')
print("\nKendall Correlation Coefficient:\n", kendall_corr)

# Covariance - Indicates the directional relationship between two variables
covariance_matrix = numerical_features.cov()
print("\nCovariance Matrix:\n", covariance_matrix)

# Correlation Matrix and Heatmap - Displays correlation coefficients between variables in a grid for easy visualization
numerical_cols = ['Number of Services', 'Number of Medicare Beneficiaries', 
                  'Number of Distinct Medicare Beneficiary/Per Day Services', 
                  'Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 
                  'Average Medicare Payment Amount', 'Average Medicare Standardized Amount']
corr_matrix = df[numerical_cols].corr()
plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Matrix Heatmap")
plt.show()

# Scatter Plot - Shows the relationship between two continuous variables
def scatterPlot(feature1, feature2):
    plt.figure(figsize=(4,3))
    sns.scatterplot(x=feature1, y=feature2)
    plt.show()
for col in numerical_cols:
    for col1 in numerical_cols:
        if col != col1:
            scatterPlot(df[col], df[col1])

# Bivariate Analysis - Numerical vs Categorical Data
numerical_features = df.select_dtypes(include=['float64', 'int64'])
categorical_features = df.select_dtypes(include=['object'])

# Box Plot
def BoxPlot(feature1, feature2):
    sns.boxplot(x=df[feature1], y=df[feature2], data=df)
    plt.title(f"Box Plot: {feature1} by {feature2}")
    plt.figure(figsize=(4,3))
    plt.xticks(rotation=90)
    plt.show()
for col in numerical_features:
    for col1 in categorical_features:
        BoxPlot(col, col1)

# Bar Plots - Displays data distribution and highlights outliers through quartiles
def BarPlot(feature1, feature2):
    plt.figure(figsize=(4, 3))
    sns.barplot(x=df[feature1], y=df[feature2] , data=df, estimator=sum)
    plt.xticks(rotation=90)
    plt.title(f"{feature1} by {feature2}")
    plt.show()
for col in numerical_features:
    for col1 in categorical_features:
        BoxPlot(col, col1)

# Bivariate Analysis - Categorical vs Categorical Data
# Count plot (Stacked Bar Plot) - Plots counts of categorical data
def CountPlot(feature1, feature2):
    sns.countplot(x=df[feature1], hue=df[feature2], data=df)
    plt.title(f"Count Plot: {feature1} vs {feature2}")
    plt.show()
for col in categorical_cols:
    CountPlot(col, 'Gender of the Provider')

# Multivariate Analysis 
# Pairplot of numerical features - Visualizes pairwise relationships and distributions across multiple variables
sns.pairplot(df[numerical_cols], diag_kind='kde')
plt.suptitle("Pairplot of Numerical Features")
plt.show()

# Grouped bar plot for categorical and numerical features - Compares subcategories within categorical groups
df.groupby(['Provider Type', 'Gender of the Provider'])['Number of Services'].mean().unstack().plot(kind='bar', figsize=(4,3))
plt.title("Mean Number of Services by Provider Type and Gender")
plt.show()