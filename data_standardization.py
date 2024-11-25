from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
import pandas as pd

class DataStandardizer:
    def __init__(self, data, numerical_columns):
        self.data = data
        self.numerical_columns = numerical_columns
        self.scalers = {}

    def compare_standardization(self):
        print("Comparing standardization techniques...")

        print("\nStandardScaler (Optional Comparison)...")
        standard_scaled = self.data[self.numerical_columns].apply(lambda x: (x - x.mean()) / x.std(), axis=0)
        print("Standard Scaled Data:\n", standard_scaled.head())

        print("\nQuantile Transformer (Optional Comparison)...")
        quantile_transformer = QuantileTransformer(output_distribution='uniform')
        quantile_transformed = quantile_transformer.fit_transform(self.data[self.numerical_columns])
        print("Quantile Transformed Data:\n", pd.DataFrame(quantile_transformed, columns=self.numerical_columns).head())

        print("\nPowerTransformer (Chosen Method - Yeo-Johnson)...")
        power_transformer = PowerTransformer(method='yeo-johnson')
        power_transformed = power_transformer.fit_transform(self.data[self.numerical_columns])
        self.scalers['power_transformer'] = power_transformer
        print("Power Transformed Data:\n", pd.DataFrame(power_transformed, columns=self.numerical_columns).head())

        return pd.DataFrame(power_transformed, columns=self.numerical_columns)

    def apply_power_transformer(self):
        if 'power_transformer' not in self.scalers:
            transformer = PowerTransformer(method='yeo-johnson')
            self.scalers['power_transformer'] = transformer
        else:
            transformer = self.scalers['power_transformer']

        self.data[self.numerical_columns] = transformer.fit_transform(self.data[self.numerical_columns])
        print(f"Columns {self.numerical_columns} transformed using PowerTransformer (Yeo-Johnson).")
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

    standardizer = DataStandardizer(data, numerical_columns)
    power_transformed_data = standardizer.compare_standardization()
    print("\nFinal Power Transformed Data for Comparison:\n", power_transformed_data.head())

    standardized_data = standardizer.apply_power_transformer()
    print("\nFinal Dataset with Power Transformation Applied:\n", standardized_data.head())
