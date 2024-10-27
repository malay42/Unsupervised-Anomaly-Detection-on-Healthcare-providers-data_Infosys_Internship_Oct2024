import pandas as pd
df = pd.read_csv("C:\\Users\\malla\\Downloads\\archive\\Healthcare Providers.csv")
missing_threshold = 0.5
df = df.loc[:, df.isnull().mean() < missing_threshold]
categorical_columns = df.select_dtypes(include=['object', 'category']).columns
for col in categorical_columns:
most_frequent = df[col].mode()[0]
df[col] = df[col].fillna(most_frequent)
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_columns:
median_value = df[col].median()
df[col] = df[col].fillna(median_value)
df = df.dropna()
df = df.drop_duplicates()
print("Remaining missing values per column:")
print(df.isnull().sum())
