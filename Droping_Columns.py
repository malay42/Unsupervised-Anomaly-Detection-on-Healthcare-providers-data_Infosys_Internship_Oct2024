# import pandas as pd

# # Load the cleaned CSV file
# file_path = 'cleaned_file.csv'
# data = pd.read_csv(file_path)

# # Show the first few rows and get a summary of the dataset to analyze its structure
# data_info = data.info()
# data_head = data.head()

# data_info, data_head


import pandas as pd

df = pd.read_csv('cleaned_file.csv')

# List of columns to drop
columns_to_drop = ['Street Address 1 of the Provider', 
                   'Street Address 2 of the Provider', 
                   'City of the Provider', 
                   'Zip Code of the Provider', 
                   'Middle Initial of the Provider', 
                   'Credentials of the Provider']

# Dropping the irrelevant columns
df.drop(columns=columns_to_drop, inplace=True)

df.to_csv('more_filtered_file.csv', index=False)

print("done'")
