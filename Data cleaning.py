#imported the required libraries for cleaning process
import pandas as pd

# Reading the CSV file
df = pd.read_csv('your_file.csv')

# Removing duplicate rows
df.drop_duplicates(inplace=True)

# Removing rows with any null values
df.dropna(inplace=True)

# Saving the cleaned data to a new CSV file
df.to_csv('cleaned_file.csv', index=False)

print("Data cleaning completed. Cleaned data saved to 'cleaned_file.csv'.")
