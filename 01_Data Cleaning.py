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