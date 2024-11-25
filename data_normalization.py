from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, Normalizer
from sklearn.preprocessing import Normalizer
import pandas as pd

class DataNormalizer:
    def __init__(self, data, numerical_columns):
        self.data = data
        self.numerical_columns = numerical_columns
        self.scalers = {}

    def compare_normalization(self):
        print("Comparing normalization techniques...")
        print("\nMinMax Normalization (Optional Comparison)...")
        min_max_normalized = self.data[self.numerical_columns].apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
        print("MinMax Normalized Data:\n", min_max_normalized.head())

        print("\nMaxAbs Normalization (Optional Comparison)...")
        max_abs_normalized = self.data[self.numerical_columns].apply(lambda x: x / x.abs().max(), axis=0)
        print("MaxAbs Normalized Data:\n", max_abs_normalized.head())

        print("\nL1 Normalization (Optional Comparison)...")
        l1_normalizer = Normalizer(norm='l1')
        l1_normalized = l1_normalizer.fit_transform(self.data[self.numerical_columns])
        print("L1 Normalized Data:\n", pd.DataFrame(l1_normalized, columns=self.numerical_columns).head())
        
        print("\nL2 Normalization (Chosen Method)...")
        l2_normalizer = Normalizer(norm='l2')
        l2_normalized = l2_normalizer.fit_transform(self.data[self.numerical_columns])
        self.scalers['l2_normalizer'] = l2_normalizer
        print("L2 Normalized Data:\n", pd.DataFrame(l2_normalized, columns=self.numerical_columns).head())

        return pd.DataFrame(l2_normalized, columns=self.numerical_columns)

    def apply_l2_normalization(self):
        if 'l2_normalizer' not in self.scalers:
            normalizer = Normalizer(norm='l2')
            self.scalers['l2_normalizer'] = normalizer
        else:
            normalizer = self.scalers['l2_normalizer']

        self.data[self.numerical_columns] = normalizer.fit_transform(self.data[self.numerical_columns])
        print(f"Columns {self.numerical_columns} normalized with L2 normalization.")
        return self.data

if __name__ == "__main__":
    data = pd.read_csv('cleaned_healthcare.csv')
    numerical_columns = [
        'Number of Services',
        'Number of Medicare Beneficiaries',
        'Number of Distinct Medicare Beneficiary/Per Day Services',
        'Average Medicare Allowed Amount',
        'Average Submitted Charge Amount',
        'Average Medicare Payment Amount',
        'Average Medicare Standardized Amount',
        'Gender of the Provider'
    ]
    for col in numerical_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data = data.dropna(subset=numerical_columns)
    normalizer = DataNormalizer(data, numerical_columns)

    l2_normalized_data = normalizer.compare_normalization()
    print("\nFinal L2 Normalized Data for Comparison:\n", l2_normalized_data.head())

    normalized_data = normalizer.apply_l2_normalization()
    print("\nFinal Dataset with L2 Normalization Applied:\n", normalized_data.head())
