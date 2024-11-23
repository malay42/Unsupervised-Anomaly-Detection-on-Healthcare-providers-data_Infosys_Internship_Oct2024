# To check whether given entry is anomaly or not

import pandas as pd
import numpy as np

df = pd.read_csv('Healthcare_Providers.csv')        # Loading dataset
df1 = pd.read_csv('Healthcare_Providers-copy.csv')        # Loading copy of dataset

# Function to get details based on index
def get_record_details(index, df, df1, anomaly_labels):
    try:
        # Get the specific row details from df1 (original dataset)
        record_details = df1.iloc[index]
        
        # Check if the record is an anomaly
        is_anomaly = "Anomaly" if autoencoder_labels[index] == 1 else "Normal"
        
        # Print the details
        print("Record Details at Index", index)
        print(record_details)
        print("Anomaly Status:", is_anomaly)
        
        return record_details, is_anomaly
    except IndexError:
        print(f"Index {index} is out of bounds. Please enter a valid index within the dataset range.")
        return None, None

# Example usage
index = int(input("Enter the index of the record to check: "))
record, anomaly_status = get_record_details(index, df, df1, autoencoder_labels)