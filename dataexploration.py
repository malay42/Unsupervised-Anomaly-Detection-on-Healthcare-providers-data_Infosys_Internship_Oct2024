#importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
#reading and loading the dataset
df=pd.read_csv('Healthcare Providers.csv')
#printing the dataset
df
#printing the data of first five rows
print(df.head())
#  summary of descriptive statistics for numeric columns
print(df.describe())
#Data types and non-null counts
print(df.info())