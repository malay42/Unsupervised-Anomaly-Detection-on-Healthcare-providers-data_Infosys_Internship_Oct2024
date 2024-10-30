import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv('Healthcare_Providers.csv')

# Understanding the Dataset
print(f"Shape of Dataset: {df.shape} \n")
print(df.head())
print(df.info())
print(df.describe())