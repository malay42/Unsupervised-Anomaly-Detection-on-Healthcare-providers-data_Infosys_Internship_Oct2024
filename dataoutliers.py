# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

file_path = 'Processed_Healthcare_Providers.csv'
df = pd.read_csv(file_path)



# %%
df.describe()


# %%
columns = df.columns.tolist()

# Print the column names
print("Columns in the dataset:")
for col in columns:
    print(col)

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
file_path = 'Processed_Healthcare_Providers.csv'
df = pd.read_csv(file_path)

# List of numerical columns to create box plots for
numerical_cols = [
    'Number of Services',
    'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services',
    'Average Medicare Allowed Amount',
    'Average Submitted Charge Amount',
    'Average Medicare Payment Amount',
    'Average Medicare Standardized Amount'
]

# Create box plots for each numerical column
for col in numerical_cols:
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=data[col])
    plt.title(f'Box Plot of {col}')
    plt.ylabel(col)
    plt.show()


# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
file_path = 'Processed_Healthcare_Providers.csv'
df = pd.read_csv(file_path)

# Identify all numerical columns in the dataset
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Create box plots for each numerical column
for col in numerical_cols:
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=data[col])
    plt.title(f'Box Plot of {col}')
    plt.ylabel(col)
    plt.show()


# %%
#Z-SCORE METHOD
import pandas as pd
import numpy as np
from scipy.stats import zscore

# Load the dataset
file_path = 'Processed_Healthcare_Providers.csv'
data = pd.read_csv(file_path)

# Function to remove outliers using the Z-score method
def remove_outliers_zscore(data, threshold=3):
    # Identify numerical columns
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns

    # Calculate Z-scores for all numerical columns
    z_scores = data[numerical_cols].apply(zscore)

    # Filter out rows where any Z-score value exceeds the specified threshold
    data_filtered = data[(np.abs(z_scores) < threshold).all(axis=1)]

    return data_filtered

# Remove outliers from the dataset using the Z-score method
data_cleaned = remove_outliers_zscore(data, threshold=3)

# Save the cleaned dataset to a new CSV file
data_cleaned.to_csv('Cleaned_Healthcare_Providers.csv', index=False)



# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the cleaned dataset
file_path = 'Cleaned_Healthcare_Providers.csv'
df = pd.read_csv(file_path)

# Identify all numerical columns in the cleaned dataset
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Create box plots for each numerical column
for col in numerical_cols:
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=df[col])
    plt.title(f'Box Plot of {col} (Outliers Removed)')
    plt.ylabel(col)
    plt.show()


# %%
import pandas as pd
file_path = 'Cleaned_Healthcare_Providers.csv'
df = pd.read_csv(file_path)


# %%
upper_limit=df['Number of Services'].mean()+3*df['Number of Services'].std()
lower_limit=df['Number of Services'].mean()-3*df['Number of Services'].std()
print('upper_limit:',upper_limit)
print('lower_limit:',lower_limit)

# %%
new_df=df.loc[(df['Number of Services']<upper_limit)&(df['Number of Services']>lower_limit)]
print('before removing outliers:', len(df))
print('after removing outliers:',len(new_df))
print('outliers',len(df)-len(new_df))

# %%
import seaborn as sns

# %%
sns.boxplot(new_df['Number of Services'])


