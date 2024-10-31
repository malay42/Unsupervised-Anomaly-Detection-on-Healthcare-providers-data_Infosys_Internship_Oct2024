import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
data = data_dropped
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

normalize_cols = [
    'Number of Services',
    'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services'
]

standardize_cols = [
    'Average Medicare Allowed Amount',
    'Average Submitted Charge Amount',
    'Average Medicare Payment Amount',
    'Average Medicare Standardized Amount'
]

min_max_scaler = MinMaxScaler()
data_dropped[normalize_cols] = min_max_scaler.fit_transform(data_dropped[normalize_cols])
scaler = StandardScaler()
data_dropped[standardize_cols] = scaler.fit_transform(data_dropped[standardize_cols])
print(data_dropped.head())