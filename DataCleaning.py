# Importing Libraries
import pandas as pd

# Define a function to load and clean healthcare provider data
def load_and_clean_data(file_path='Healthcare Providers.csv'):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Print the shape and null value count before cleaning
    print("Before Cleaning:")
    print("Shape:", df.shape)
    print("Missing Values:")
    print(df.isnull().sum()) 

    # Drop unnecessary columns
    df = df.drop(columns=['Street Address 2 of the Provider', 'Middle Initial of the Provider'])
    
    # Remove rows where 'First Name of the Provider' is missing (NaN)
    df = df.dropna(subset=['First Name of the Provider'], how='all')
    
    # Fill missing values in 'Gender of the Provider' column with the most frequent gender
    most_frequent_gender = df['Gender of the Provider'].mode()[0]
    df['Gender of the Provider'] = df['Gender of the Provider'].fillna(most_frequent_gender)
    
    # Remove rows where 'Credentials of the Provider' is missing (NaN)
    df = df.dropna(subset=['Credentials of the Provider'], how='all')
    
    # Remove duplicate rows
    df = df.drop_duplicates()

    # Print the shape and null value count after cleaning
    print()
    print("After Cleaning:")
    print("Shape:", df.shape)
    print("Missing Values:")
    print(df.isnull().sum())

    return df

if __name__ == "__main__":
    df = load_and_clean_data() 

    # Create a new file with the cleaned data
    df.to_csv('Cleaned_Healthcare_Providers.csv', index=False)
