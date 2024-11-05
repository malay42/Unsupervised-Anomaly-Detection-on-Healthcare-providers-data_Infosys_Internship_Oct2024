import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
df=pd.read_csv('Healthcare Providers.csv')
df
print(df.head())
print(df.describe())
print(df.info())