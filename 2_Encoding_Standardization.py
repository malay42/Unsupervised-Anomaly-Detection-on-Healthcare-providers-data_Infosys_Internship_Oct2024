# Importing Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, Normalizer, MinMaxScaler
from sklearn.pipeline import Pipeline

# Function Definitions
def load_data(file_path):
    """Load the CSV file into a DataFrame."""
    return pd.read_csv(file_path)

def remove_comma(value):
    """Remove commas from a given string."""
    return value.replace(",", "")

def process_numerical_columns(df, columns):
    """Convert specified columns to numeric after removing commas."""
    for column in columns:
        df[column] = pd.to_numeric(df[column].apply(remove_comma))
    return df

def encode_categorical_columns(df, columns):
    """Encode specified categorical columns using LabelEncoder."""
    label_encoder = LabelEncoder()
    for column in columns:
        df[f"{column}_label"] = label_encoder.fit_transform(df[column].astype(str))
    return df

def save_data(df, file_name):
    """Save DataFrame to a CSV file."""
    df.to_csv(file_name, index=False)

def plot_pairgrid(data, columns, title, sample_size=600):
    """Create and display a PairGrid plot for the given columns."""
    sampled_data = data[columns].sample(sample_size, random_state=42)
    g = sns.PairGrid(sampled_data)
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle(title)
    g.map_diag(sns.kdeplot)
    g.map_offdiag(sns.kdeplot)
    plt.show(block=True)

def normalize_and_scale_data(data, pipeline):
    """Apply normalization and scaling using the provided pipeline."""
    return pipeline.fit_transform(data)

# Main execution
if __name__ == "__main__":
    # Define file paths and column names
    file_path = "Healthcare_Providers.csv"
    output_file = "data_encoded.csv"
    numerical_columns = [
        "Number of Services", "Number of Medicare Beneficiaries",
        "Number of Distinct Medicare Beneficiary/Per Day Services",
        "Average Medicare Allowed Amount", "Average Submitted Charge Amount",
        "Average Medicare Payment Amount", "Average Medicare Standardized Amount"
    ]
    categorical_columns = [
        "Credentials of the Provider", "Gender of the Provider",
        "Entity Type of the Provider", "State Code of the Provider",
        "Country Code of the Provider", "Provider Type",
        "Medicare Participation Indicator", "Place of Service",
        "HCPCS Code", "HCPCS Description", "HCPCS Drug Indicator"
    ]
    pairgrid_columns = [
        "Number of Services", "Average Medicare Allowed Amount",
        "Average Submitted Charge Amount", "Average Medicare Payment Amount"
    ]

    # Load and preprocess data
    df = load_data(file_path)
    df = process_numerical_columns(df, numerical_columns)
    df = encode_categorical_columns(df, categorical_columns)

    # Select encoded and numerical data for further processing
    selected_data = df[
        [f"{col}_label" for col in categorical_columns] + numerical_columns
    ]

    # Save the processed data
    save_data(selected_data, output_file)

    # Plot PairGrid before transformation
    plot_pairgrid(selected_data, pairgrid_columns, title="Before Normalization:")

    # Configure normalization and scaling pipeline
    processing_pipeline = Pipeline([
        ("normalizer", Normalizer()),
        ("scaler", MinMaxScaler())
    ])

    # Apply pipeline transformations
    transformed_data = normalize_and_scale_data(selected_data, processing_pipeline)

    # Convert transformed data back to a DataFrame
    transformed_df = pd.DataFrame(
        transformed_data, columns=selected_data.columns
    )

    # Plot PairGrid after transformation
    plot_pairgrid(transformed_df, pairgrid_columns, title="After Normalization:")
