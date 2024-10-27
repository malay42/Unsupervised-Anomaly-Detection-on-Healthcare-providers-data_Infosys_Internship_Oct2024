# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce

# Load the cleaned data
cleaned_file_path = "C:/Users/shiva/Desktop/PROJECT 1/Cleaned_Healthcare_Providers.csv"
df = pd.read_csv(cleaned_file_path)

# Label Encoding for 'Gender of the Provider' with visualization
if 'Gender of the Provider' in df.columns:
    plt.figure(figsize=(12, 5))
    
    # Before encoding
    plt.subplot(1, 2, 1)
    sns.countplot(x='Gender of the Provider', data=df)
    plt.title("Gender of the Provider (Before Encoding)")
    
    # Apply label encoding
    label_encoder = LabelEncoder()
    df['Gender of the Provider Encoded'] = label_encoder.fit_transform(df['Gender of the Provider'])
    
    # After encoding
    plt.subplot(1, 2, 2)
    sns.histplot(df['Gender of the Provider Encoded'], bins=3, kde=False)
    plt.title("Gender of the Provider (After Encoding)")
    plt.xlabel("Encoded Gender Value")
    
    plt.tight_layout()
    plt.show()
    print("Label Encoding completed for column: Gender of the Provider")

# Binary Encoding for 'Provider Type' with visualization
binary_encoding_columns = ['Provider Type']
if 'Provider Type' in df.columns:
    binary_encoder = ce.BinaryEncoder(cols=binary_encoding_columns)
    df_binary_encoded = binary_encoder.fit_transform(df)
    print(f"Binary Encoding completed for columns: {binary_encoding_columns}")
    
    # Visualize binary encoding output for 'Provider Type'
    plt.figure(figsize=(10, 6))
    binary_columns = [col for col in df_binary_encoded.columns if 'Provider Type' in col]
    df_binary_encoded[binary_columns].sum().plot(kind='bar', color='skyblue')
    plt.title('Binary Encoding Distribution for Provider Type')
    plt.xlabel('Binary Encoded Columns')
    plt.ylabel('Sum of Encoded Values')
    plt.show()
else:
    print("Column 'Provider Type' not found in the dataset.")

# One-Hot Encoding for 'Gender of the Provider' with visualization
if 'Gender of the Provider' in df.columns:
    df_one_hot = pd.get_dummies(df, columns=['Gender of the Provider'], prefix='Gender')
    print("One-Hot Encoding completed for column: Gender of the Provider")
    
    # Visualize one-hot encoding output for 'Gender of the Provider'
    plt.figure(figsize=(8, 5))
    one_hot_columns = [col for col in df_one_hot.columns if 'Gender_' in col]
    df_one_hot[one_hot_columns].sum().plot(kind='bar', color='lightgreen')
    plt.title('One-Hot Encoding Distribution for Gender')
    plt.xlabel('One-Hot Encoded Gender Columns')
    plt.ylabel('Count')
    plt.show()

# Frequency Encoding for 'Provider Type' with visualization
frequency_encoding_column = 'Provider Type'
if frequency_encoding_column in df.columns:
    freq_encoding = df[frequency_encoding_column].value_counts().to_dict()
    df[frequency_encoding_column + '_Freq'] = df[frequency_encoding_column].map(freq_encoding)
    print("Frequency Encoding completed for column:", frequency_encoding_column)
    
    # Visualize Frequency Encoding for 'Provider Type'
    plt.figure(figsize=(10, 5))
    sns.histplot(df[frequency_encoding_column + '_Freq'], kde=True, bins=30, color='salmon')
    plt.title('Frequency Encoding for Provider Type')
    plt.xlabel('Frequency Encoded Values')
    plt.ylabel('Count')
    plt.show()
else:
    print(f"Column '{frequency_encoding_column}' does not exist in the dataset. Skipping frequency encoding.")
