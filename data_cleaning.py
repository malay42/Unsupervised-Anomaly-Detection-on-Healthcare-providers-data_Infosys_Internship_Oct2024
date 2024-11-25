
import pandas as pd

class DataCleaner:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        
    def drop_columns(self, columns_to_drop):
        self.data.drop(columns=columns_to_drop, axis=1, inplace=True)
        
    def fill_categorical_missing(self, categorical_columns):
        for column in categorical_columns:
            mode_value = self.data[column].mode()[0]
            self.data[column].fillna(mode_value, inplace=True)
            
    def fill_numerical_missing(self, numerical_columns):
        for column in numerical_columns:
            median_value = self.data[column].median()
            self.data[column].fillna(median_value, inplace=True)
            
    def round_columns(self, columns_to_round):
        for column in columns_to_round:
            self.data[column] = pd.to_numeric(self.data[column], errors='coerce').round(2)
            
    def encode_categorical(self, encoding_map):
        for column, mapping in encoding_map.items():
            self.data[column] = self.data[column].map(mapping)
            
    def fill_remaining_missing(self, fill_value=0):
        self.data.fillna(fill_value, inplace=True)
        
    def save_cleaned_data(self, output_path):
        self.data.to_csv(output_path, index=False)
        
    def get_data(self):
        return self.data


if __name__ == "__main__":
    cleaner = DataCleaner("Healthcare Providers.csv")
    cleaner.drop_columns(['index', 'Street Address 2 of the Provider', 'Middle Initial of the Provider', 'First Name of the Provider'])
    cleaner.fill_categorical_missing(['Last Name/Organization Name of the Provider', 'Credentials of the Provider', 'Gender of the Provider'])
    cleaner.fill_numerical_missing(['Zip Code of the Provider'])
    cleaner.round_columns(['Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 
                           'Average Medicare Payment Amount', 'Average Medicare Standardized Amount'])
    cleaner.encode_categorical({
        'Gender of the Provider': {'F': 1, 'M': 0},
        'Entity Type of the Provider': {'I': 1, 'O': 0}
    })
    cleaner.fill_remaining_missing(0)
    cleaner.save_cleaned_data("Healthcare_Providers_Cleaned.csv")
