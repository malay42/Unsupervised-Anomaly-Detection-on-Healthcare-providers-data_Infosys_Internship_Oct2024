import pandas as pd 
import numpy as np 
import seaborn as sn
import matplotlib.pyplot as plt

df = pd.read_csv('Cleaned_HealthCare.csv')

#Bar Graph to show relation between Average Submitted Charge Amount and Gender of the Provider
plt.figure(figsize=(10,8))
sn.barplot(x ='Gender of the Provider', y = 'Average Submitted Charge Amount', data=df, estimator='mean')
plt.title('Average Submitted Charge Amount by Gender of the Provider')
plt.xlabel('Gender of the Provider')
plt.ylabel('Average Submitted Charge Amount')
plt.show()

#Bar Graph to show relation between Total Number of Services and Entity Type of Provider
plt.figure(figsize=(6, 4))
sn.barplot(x='Entity Type of the Provider', y='Number of Services', data=df, estimator='sum')
plt.title('Total Number of Services by Entity Type of Provider')
plt.xlabel('Entity Type of Provider')
plt.ylabel('Total Number of Services')
plt.show()

#Bar Graph to show relation between Total Number of Services and HCPCS Drug Indicator
plt.figure(figsize=(6, 4))
sn.barplot(x='HCPCS Drug Indicator', y='Number of Services', data=df, estimator='sum')
plt.title('Total Number of Services by HCPCS Drug Indicator')
plt.xlabel('HCPCS Drug Indicator')
plt.ylabel('Total Number of Services')
plt.show()

#Pie Chart to show top 10 Credentials of the Provider
top_credentials = df['Credentials of the Provider'].value_counts().nlargest(10)
plt.figure(figsize=(8, 8))
plt.pie(top_credentials, labels=top_credentials.index, autopct='%1.1f%%', startangle=140)
plt.title('Top 10 Credentials of Providers')
plt.axis('equal')  
plt.show()

#Correlation Heatmap
plt.figure(figsize=(12, 8))
corr = df[['Number of Services', 'Number of Medicare Beneficiaries', 
             'Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 
             'Average Medicare Payment Amount', 'Average Medicare Standardized Amount']].corr()

sn.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

#Scatter Plot to show relation between Average Submitted Charge and Average Medicare Payment
plt.figure(figsize=(10, 6))
sn.scatterplot(data=df, x='Average Submitted Charge Amount', y='Average Medicare Payment Amount',hue='Gender of the Provider')
plt.title('Average Submitted Charge vs. Average Medicare Payment')
plt.show()

#Bar Graph to show top 10 HCPCS Codes
plt.figure(figsize=(4, 4))
top_hcpcs = df['HCPCS Code'].value_counts().nlargest(10)
sn.barplot(x=top_hcpcs.index, y=top_hcpcs.values)
plt.title('Top 10 HCPCS Codes')
plt.ylabel('Frequency')
plt.xlabel('HCPCS Code')
plt.show()

#Bar Graph to show top 10 Cities by Number of Providers
plt.figure(figsize=(12, 6))
top_cities = df['City of the Provider'].value_counts().nlargest(10)
sn.barplot(x=top_cities.index, y=top_cities.values)
plt.title('Top 10 Cities by Number of Providers')
plt.ylabel('Number of Providers')
plt.xlabel('City')
plt.xticks(rotation=45)
plt.show()

#Pair Plot to show relation between different numerical values
sn.pairplot(df[['Number of Services', 'Number of Medicare Beneficiaries', 'Average Medicare Allowed Amount', 
                   'Average Submitted Charge Amount', 'Average Medicare Payment Amount']])
plt.show()
