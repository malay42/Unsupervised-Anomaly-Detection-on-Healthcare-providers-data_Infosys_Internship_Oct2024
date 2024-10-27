import pandas as pd

# Load dataset
df = pd.read_csv("C:\\Users\\malla\\Downloads\\archive\\Healthcare Providers.csv")

# Step 1: Drop columns with excessive missing values (e.g., more than 50% missing)
missing_threshold = 0.5
df = df.loc[:, df.isnull().mean() < missing_threshold]

# Step 2: Fill missing values in categorical columns with the most frequent value (mode)
categorical_columns = df.select_dtypes(include=['object', 'category']).columns
for col in categorical_columns:
    most_frequent = df[col].mode()[0]
    df[col] = df[col].fillna(most_frequent)

# Step 3: Fill missing values in numerical columns with the median value
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_columns:
    median_value = df[col].median()
    df[col] = df[col].fillna(median_value)

# Step 4: Drop rows with any remaining NaN values if necessary
df = df.dropna()

# Step 5: Drop duplicate rows to ensure each record is unique
df = df.drop_duplicates()

# Step 6: Check for remaining missing values to confirm they’re all handled
print("Remaining missing values per column:")
print(df.isnull().sum())
