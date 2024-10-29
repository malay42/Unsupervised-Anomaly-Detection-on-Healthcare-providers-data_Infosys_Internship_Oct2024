import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings. filterwarnings('ignore')

df = pd.read_csv('Healthcare Providers.csv')
print(df)

print(df.columns)
print(df.info())
print(df.isnull().sum())


DropCols = ['index', 'National Provider Identifier',
       'Last Name/Organization Name of the Provider',
       'First Name of the Provider', 'Middle Initial of the Provider','Street Address 1 of the Provider',
       'Street Address 2 of the Provider','Zip Code of the Provider',"HCPCS Code"]

df = df.drop(DropCols, axis = 1)

print(df.isnull().sum())

print(df.head())

print(df["Entity Type of the Provider"].value_counts())


def RemoveComma(x):
    return x.replace(",","")

df["Average Medicare Allowed Amount"] = pd.to_numeric(df["Average Medicare Allowed Amount"].apply(lambda x: RemoveComma(x)),
                                                             errors= "ignore")
df["Average Submitted Charge Amount"] = pd.to_numeric(df["Average Submitted Charge Amount"].apply(lambda x: RemoveComma(x)),
                                                       errors = "ignore")
df["Average Medicare Payment Amount"] = pd.to_numeric(df["Average Medicare Payment Amount"].apply(lambda x: RemoveComma(x)),
                                                       errors = "ignore")
df["Average Medicare Standardized Amount"] = pd.to_numeric(df["Average Medicare Standardized Amount"].apply(lambda x: RemoveComma(x)),
                                                             errors = "ignore")

print(df)

print(df.info())

df["Credentials of the Provider"] = df["Credentials of the Provider"].fillna(df["Credentials of the Provider"].mode()[0])
df["Gender of the Provider"] = df["Gender of the Provider"].fillna(df["Gender of the Provider"].mode()[0])

print(df.isnull().sum())

# from sklearn.preprocessing import CategoricalEncoder
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
import pandas as pd

def Preprocessing(df):
    # List of columns that may have high cardinality (many unique values)
    high_cardinality_cols = [var for var in df.columns if df[var].dtype == "O" and df[var].nunique() > 100]  # Threshold of 100 unique values, adjust as needed
    
    # Perform Binary Encoding for high cardinality columns
    encoder = ce.BinaryEncoder(cols=high_cardinality_cols)
    df = encoder.fit_transform(df)
    
    # For remaining categorical columns, apply One-Hot Encoding
    OHEcols = [var for var in df.columns if df[var].dtype == "O"]
    df = pd.get_dummies(df, columns=OHEcols, drop_first=True)
    
    # Standardization
    df_columns = df.columns
    std = StandardScaler()
    df = std.fit_transform(df)
    df = pd.DataFrame(df, columns=df_columns)
    
    return df

# Example usage
df_one1 = Preprocessing(df)
print(df_one1)


# import category_encoders as ce
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

def Preprocessing(df):
    # Store number of columns before encoding
    original_columns = df.shape[1]
    
    # List of columns that may have high cardinality (many unique values)
    high_cardinality_cols = [var for var in df.columns if df[var].dtype == "O" and df[var].nunique() > 100]  # Threshold of 100 unique values
    
    # Perform Binary Encoding for high cardinality columns
    encoder = ce.BinaryEncoder(cols=high_cardinality_cols)
    df = encoder.fit_transform(df)
    
    # For remaining categorical columns, apply One-Hot Encoding
    OHEcols = [var for var in df.columns if df[var].dtype == "O"]
    df = pd.get_dummies(df, columns=OHEcols, drop_first=True)
    
    # Store number of columns after encoding
    after_encoding_columns = df.shape[1]
    
    # Standardization
    df_columns = df.columns
    std = StandardScaler()
    df = std.fit_transform(df)
    df = pd.DataFrame(df, columns=df_columns)
    
    # Plot the comparison of columns before and after encoding
    plt.figure(figsize=(6,4))
    plt.bar(['Before Encoding', 'After Encoding'], [original_columns, after_encoding_columns], color=['skyblue', 'lightgreen'])
    plt.title('Number of Features Before and After Encoding')
    plt.ylabel('Number of Features')
    plt.show()
    
    return df

# Example usage with df
df_one1 = Preprocessing(df)