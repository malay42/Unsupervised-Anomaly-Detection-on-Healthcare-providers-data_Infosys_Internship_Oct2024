# Data Encoding & Normalization/Standarization
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('Healthcare_Providers.csv')        # Loading dataset

# Frequency Encoding - assigns values based on the frequency of each category in the dataset
frequency_features = ['Credentials of the Provider',
                      'City of the Provider',
                      'State Code of the Provider',
                      'Provider Type', 'HCPCS Code',
                      'HCPCS Description']
for feature in frequency_features:
    freq_encoding = df[feature].value_counts().to_dict()
    df[f'{feature}_enc'] = df[feature].map(freq_encoding)

# One Hot Encoding - for nominal variables(no order) and finite numer of unique values 
ohe = OneHotEncoder(sparse_output=False, drop='first')
gender_ohe = ohe.fit_transform(df[['Gender of the Provider']])
gender_ohe_df = pd.DataFrame(gender_ohe, columns=ohe.get_feature_names_out(['Gender of the Provider']))
gender_ohe_df.index = df.index
df = pd.concat([df, gender_ohe_df], axis=1)         # Concatenate the new One-Hot Encoded columns with the original DataFrame

# Binary Encoding - converts categories into binary digits
df['Entity_Type_enc'] = df['Entity Type of the Provider'].apply(lambda x: 1 if x == 'I' else 0)     # Entity Type of the Provider – Binary (I or O)
df['Medicare_Participation_enc'] = df['Medicare Participation Indicator'].apply(lambda x: 1 if x == 'y' else 0)     # Medicare Participation Indicator – Binary (y or n)
df['Place_of_Service_enc'] = df['Place of Service'].apply(lambda x: 1 if x == 'F' else 0)           # Place of Service – Binary (Facility(F) or NonFacility(O))
df['HCPCS_Drug_Indicator_enc'] = df['HCPCS Drug Indicator'].apply(lambda x: 1 if x == 'y' else 0)   # HCPCS Drug Indicator – Binary Encoding (y or n)

# Dropping the original columns
df.drop(['Credentials of the Provider', 'City of the Provider',
         'State Code of the Provider',  'Provider Type', 'HCPCS Code',
         'HCPCS Description',           'Gender of the Provider',
         'Entity Type of the Provider', 'Medicare Participation Indicator',
         'Place of Service',            'HCPCS Drug Indicator'],
          axis=1, inplace=True)

# Normalization & Standarization
# Z-Score Columns - The values are normalized to a mean of 0 and standard deviation of 1
scaler = StandardScaler() 
zscore_cols = ['Credentials of the Provider_enc',        # for features with high frequency   
               'City of the Provider_enc',
               'State Code of the Provider_enc',
               'Provider Type_enc', 'HCPCS Code_enc',
               'HCPCS Description_enc']   
df[zscore_cols] = scaler.fit_transform(df[zscore_cols])

zscore_outliers = ['Number of Services',                # for features with normal distribution
                   'Average Submitted Charge Amount',
                   'Average Medicare Payment Amount']
df[zscore_outliers] = scaler.fit_transform(df[zscore_outliers])
df[zscore_outliers].head()

# Min-Max Normalization - values are scaled down from 0 to 1 
scaler = MinMaxScaler()
minMax_cols = ['Number of Medicare Beneficiaries', 
               'Number of Distinct Medicare Beneficiary/Per Day Services', 
               'Average Medicare Allowed Amount',
               'Average Medicare Standardized Amount']
df[minMax_cols] = scaler.fit_transform(df[minMax_cols])
df[minMax_cols].head()

# Rounding all the values upto 4 decimal places
df = df.round(4)