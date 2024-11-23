# Data Encoding & Normalization/Standarization
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('Healthcare_Providers.csv')        # Loading dataset

# Frequency Encoding
frequency_features = ['Credentials of the Provider',
                      'City of the Provider',
                      'State Code of the Provider',
                      'Provider Type', 'HCPCS Code']
for feature in frequency_features:
    freq_encoding = df[feature].value_counts().to_dict()
    df[f'{feature}_enc'] = df[feature].map(freq_encoding)

# One Hot Encoding
ohe = OneHotEncoder(sparse_output=False, drop='first')
gender_ohe = ohe.fit_transform(df[['Gender of the Provider']])
gender_ohe_df = pd.DataFrame(gender_ohe, columns=ohe.get_feature_names_out(['Gender of the Provider']))
gender_ohe_df.index = df.index
df = pd.concat([df, gender_ohe_df], axis=1)         # Concatenate the new One-Hot Encoded columns with the original DataFrame

# Binary Encoding 
df['Entity_Type_enc'] = df['Entity Type of the Provider'].apply(lambda x: 1 if x == 'I' else 0)     # Entity Type of the Provider – Binary (I or O)
df['Place_of_Service_enc'] = df['Place of Service'].apply(lambda x: 1 if x == 'F' else 0)           # Place of Service – Binary (Facility(F) or NonFacility(O))
df['HCPCS_Drug_Indicator_enc'] = df['HCPCS Drug Indicator'].apply(lambda x: 1 if x == 'y' else 0)   # HCPCS Drug Indicator – Binary Encoding (y or n)

# Dropping the original columns
df.drop(['Credentials of the Provider', 'City of the Provider',
         'State Code of the Provider',  'Provider Type', 'HCPCS Code',
         'Gender of the Provider',      'Entity Type of the Provider',
         'Place of Service',            'HCPCS Drug Indicator'],
          axis=1, inplace=True)

# Normalization & Standarization
# Z-Score Columns -
scaler = StandardScaler() 
zscore_cols = ['Credentials of the Provider_enc',
               'City of the Provider_enc',
               'State Code of the Provider_enc',
               'Provider Type_enc', 'HCPCS Code_enc']   
df[zscore_cols] = scaler.fit_transform(df[zscore_cols])

zscore_outliers = ['Number of Services', 'Average Submitted Charge Amount']
df[zscore_outliers] = scaler.fit_transform(df[zscore_outliers])
df[zscore_outliers].head()

# Min-Max Normalization
scaler = MinMaxScaler()
minMax_cols = ['Number of Medicare Beneficiaries', 
               'Average Medicare Allowed Amount']
df[minMax_cols] = scaler.fit_transform(df[minMax_cols])
df[minMax_cols].head()

# Rounding all the values upto 4 decimal places
df = df.round(4)