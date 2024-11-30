import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import skew
import plotly.express as px
import plotly.offline as py
from bubbly.bubbly import bubbleplot
from sklearn.feature_extraction.text import TfidfVectorizer
data = pd.read_csv('Healthcare Providers.csv')
df = pd.DataFrame(data)
data.shape
data.head(10)
data = data.drop(columns=['Middle Initial of the Provider', 
                          'Street Address 1 of the Provider', 
                          'Street Address 2 of the Provider'])

# Display the updated DataFrame
print(data.head())
data.isnull()
data.isnull().sum()
df = pd.DataFrame(data)
#zscore
# Select only the columns of interest for outlier detection
selected_columns = [
    'Number of Services', 'Number of Medicare Beneficiaries', 
    'Number of Distinct Medicare Beneficiary/Per Day Services',
    'Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 
    'Average Medicare Payment Amount', 'Average Medicare Standardized Amount'
]

# Check for non-numeric values
print("Non-numeric values in each column:")
for column in selected_columns:
    non_numeric = df[column][~df[column].apply(lambda x: isinstance(x, (int, float)))]
    if not non_numeric.empty:
        print(f"{column}: {non_numeric.tolist()}")

# Clean the selected columns by converting to numeric
df_selected = df[selected_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values
df_selected = df_selected.dropna()

# Z-Score method for detecting outliers
z_scores = np.abs(stats.zscore(df_selected))
df_no_outliers_z = df_selected[(z_scores <= 3).all(axis=1)]  # Keep only non-outliers

# --- Visualization ---

# Boxplot for Z-score method (highlighting removed outliers)
plt.figure(figsize=(10, 8))
for i, column in enumerate(df_selected.columns, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(y=df_selected[column], color='skyblue', fliersize=0)  # Original data
    sns.scatterplot(y=df_selected.loc[(z_scores > 3)[column], column], x=df_selected.index[(z_scores > 3)[column]], 
                    color='red', marker='x', label='Outliers')  # Outliers in red
    plt.title(f"Z-score Method: {column}")

plt.tight_layout()
plt.show()
#iqr
# Select only the columns of interest for outlier detection
selected_columns = [
    'Number of Services', 'Number of Medicare Beneficiaries', 
    'Number of Distinct Medicare Beneficiary/Per Day Services',
    'Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 
    'Average Medicare Payment Amount', 'Average Medicare Standardized Amount'
]

# Check for non-numeric values
print("Non-numeric values in each column:")
for column in selected_columns:
    non_numeric = df[column][~df[column].apply(lambda x: isinstance(x, (int, float)))]
    if not non_numeric.empty:
        print(f"{column}: {non_numeric.tolist()}")

# Clean the selected columns by converting to numeric
df_selected = df[selected_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values
df_selected = df_selected.dropna()

# IQR method for detecting outliers
Q1 = df_selected.quantile(0.25)
Q3 = df_selected.quantile(0.65)
IQR = Q3 - Q1

# Define outliers based on IQR
outliers_iqr = df_selected[((df_selected < (Q1 - 1.5 * IQR)) | (df_selected > (Q3 + 1.5 * IQR))).any(axis=1)]
df_no_outliers_iqr = df_selected[~df_selected.index.isin(outliers_iqr.index)]  # Keep only non-outliers

# --- Visualization ---

# Boxplot for IQR method (highlighting removed outliers)
plt.figure(figsize=(10, 8))
for i, column in enumerate(df_selected.columns, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(y=df_selected[column], color='skyblue', fliersize=0)  # Original data
    sns.scatterplot(y=outliers_iqr[column], x=outliers_iqr.index, color='orange', marker='x', label='Outliers')  # Outliers in orange
    plt.title(f"IQR Method: {column}")

plt.tight_layout()
plt.show()


# Select only the columns of interest for outlier detection
selected_columns = [
    'Number of Services', 'Number of Medicare Beneficiaries', 
    'Number of Distinct Medicare Beneficiary/Per Day Services',
    'Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 
    'Average Medicare Payment Amount', 'Average Medicare Standardized Amount'
]

# Check for non-numeric values
print("Non-numeric values in each column:")
for column in selected_columns:
    non_numeric = df[column][~df[column].apply(lambda x: isinstance(x, (int, float)))]
    if not non_numeric.empty:
        print(f"{column}: {non_numeric.tolist()}")

# Clean the selected columns by converting to numeric
df_selected = df[selected_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values
df_selected = df_selected.dropna()

# Winsorization function
def winsorize_series(series, lower_percentile=0.01, upper_percentile=0.99):
    lower_bound = series.quantile(lower_percentile)
    upper_bound = series.quantile(upper_percentile)
    return np.where(series < lower_bound, lower_bound, np.where(series > upper_bound, upper_bound, series))

# Apply Winsorization to the selected columns
df_winsorized = df_selected.copy()
for column in df_selected.columns:
    df_winsorized[column] = winsorize_series(df_selected[column])

# --- Visualization ---

# Boxplots for Winsorized data (to compare before and after)
plt.figure(figsize=(10, 8))
for i, column in enumerate(df_selected.columns, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(y=df_selected[column], color='skyblue', fliersize=0)  # Original data
    sns.boxplot(y=df_winsorized[column], color='lightgreen')  # Winsorized data
    plt.title(f"Before & After Winsorization: {column}")

plt.tight_layout()
plt.show()
z_scores = np.abs(stats.zscore(df_selected))
outliers_z = df_selected[(z_scores > 3).any(axis=1)]

# --- IQR method for detecting outliers ---
Q1 = df_selected.quantile(0.25)
Q3 = df_selected.quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = df_selected[((df_selected < (Q1 - 1.5 * IQR)) | (df_selected > (Q3 + 1.5 * IQR))).any(axis=1)]

# --- Winsorization ---
def winsorize_series(series, lower_percentile=0.01, upper_percentile=0.99):
    lower_bound = series.quantile(lower_percentile)
    upper_bound = series.quantile(upper_percentile)
    return np.where(series < lower_bound, lower_bound, np.where(series > upper_bound, upper_bound, series))

df_winsorized = df_selected.copy()
for column in df_selected.columns:
    df_winsorized[column] = winsorize_series(df_selected[column])

# Count the number of outliers detected by each method
num_outliers_z = len(outliers_z)
num_outliers_iqr = len(outliers_iqr)
num_outliers_winsor = len(df_selected) - len(df_winsorized)  # Winsorization reduces extreme values, doesn't drop rows

# --- Visualization ---

# Bar chart comparing the number of outliers detected by each method
methods = ['Z-score', 'IQR', 'Winsorization']
num_outliers = [num_outliers_z, num_outliers_iqr, num_outliers_winsor]

plt.figure(figsize=(8, 6))
sns.barplot(x=methods, y=num_outliers, palette='coolwarm')
plt.xlabel('Outlier Detection Method')
plt.ylabel('Number of Outliers Detected')
plt.title('Comparison of Outlier Detection Techniques')
plt.show()

# Boxplot to compare data distributions after each technique
plt.figure(figsize=(12, 8))
for i, column in enumerate(df_selected.columns, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(data=[df_selected[column], df_selected[column].loc[df_selected.index.difference(outliers_z.index)], 
                      df_selected[column].loc[df_selected.index.difference(outliers_iqr.index)], df_winsorized[column]], 
                palette='coolwarm', showmeans=True)
    plt.title(f"Comparison for {column}")
    plt.xticks([0, 1, 2, 3], ['Original', 'Z-Score', 'IQR', 'Winsorized'])

plt.tight_layout()
plt.show()
# Calculate remaining data points after each method
total_data_points = len(df_selected)
remaining_z_score = len(df_selected) - num_outliers_z
remaining_iqr = len(df_selected) - num_outliers_iqr
remaining_winsorization = len(df_winsorized)

# Data for the funnel chart
stages = ['Total Data Points', 'After Z-Score', 'After IQR', 'After Winsorization']
values = [total_data_points, remaining_z_score, remaining_iqr, remaining_winsorization]

# Create the funnel chart
fig = go.Figure(go.Funnel(
    y=stages,
    x=values,
    textinfo="value+percent initial",
    marker={"color": ["deepskyblue", "orange", "lightgreen", "lightblue"]}
))

# Update the layout for better presentation
fig.update_layout(
    title="Funnel Chart for Outlier Detection Techniques",
    funnelmode="stack",
    width=700,
    height=500
)

fig.show()
# Select only the columns of interest for normalization
selected_columns = [
    'Number of Services', 'Number of Medicare Beneficiaries', 
    'Number of Distinct Medicare Beneficiary/Per Day Services',
    'Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 
    'Average Medicare Payment Amount', 'Average Medicare Standardized Amount'
]

# Clean the selected columns by converting to numeric
df_selected = df[selected_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values
df_selected = df_selected.dropna()

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the data
df_normalized_minmax = pd.DataFrame(scaler.fit_transform(df_selected), columns=df_selected.columns)

# Display the normalized data
print(df_normalized_minmax)
most_frequent_gender = data['Gender of the Provider'].mode()[0]
data['Gender of the Provider'].fillna(most_frequent_gender, inplace=True)
data.isnull().sum()
# Numerical columns
numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
print("Numerical Columns:")
print(numerical_columns)

# Categorical columns 
categorical_columns = data.select_dtypes(include=['object', 'category', 'bool']).columns
print("\nCategorical Columns:")
print(categorical_columns)
# Select numerical columns (float64, int64)
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns

# Calculate skewness for each numerical column
for column in numerical_columns:
    skewness_pandas = data[column].skew()
    print(f"Skewness of {column}:", skewness_pandas)
#data preprocessing
label_encoder = LabelEncoder()
data['Credentials of the Provider'] = label_encoder.fit_transform(data['Credentials of the Provider'])
data['Credentials of the Provider']
# Initialize Label Encoders
gender_encoder = LabelEncoder()
entity_type_encoder = LabelEncoder()
city_encoder = LabelEncoder()
drug_indicator_encoder = LabelEncoder()
# Apply Label Encoding to the columns
data['Gender of the Provider'] = gender_encoder.fit_transform(data['Gender of the Provider'])
data['Entity Type of the Provider'] = entity_type_encoder.fit_transform(data['Entity Type of the Provider'])
data['City of the Provider'] = city_encoder.fit_transform(data['City of the Provider'])
data['HCPCS Drug Indicator'] = drug_indicator_encoder.fit_transform(data['HCPCS Drug Indicator'])


# Display the updated DataFrame
print(data)
sns.pairplot(data)
plt.show()
numeric_data = data.select_dtypes(include=[np.number])
# Now generate the heatmap
sns.heatmap(numeric_data.corr(), annot=True, cmap='copper')
plt.title('Correlation Heatmap', fontsize=15)
plt.show()
# Plot distplot for each numerical column in the dataset
numeric_data = data.select_dtypes(include=[np.number])

for col in numeric_data.columns:
    plt.figure()
    sns.histplot(data[col], kde=True, color='blue')
    plt.title(f'Distribution Plot of {col}', fontsize=15)
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.show()


from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
#pip install bubbly
from ipywidgets import interact_manual
import plotly.express as px
def plot_distribution(column):
    plt.figure(figsize=(10, 5))
    sns.histplot(df[column], kde=True, bins=20)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()
numerical_columns = data.select_dtypes(include='number').columns.tolist()
df1 = ['Last Name/Organization Name of the Provider','First Name of the Provider']
df1 = [col for col in df1 if col in data.columns and data[col].dtype == 'object']
def plot_categorical_distribution(column):
    if column in data.columns:
        plt.figure(figsize=(10, 5))
        sns.countplot(data=data, x=column, order=data[column].value_counts().index)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.show()
    else:
        print(f"Column {column} not found in dataframe")

from ipywidgets import interact_manual

interact_manual(plot_categorical_distribution, column=numerical_columns)

interact(plot_categorical_distribution, column=df1)
@interact_manual
def viz(x=list(data.select_dtypes('number').columns[1:]), y=list(data.select_dtypes('number').columns[1:])):
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=data[x], y=data[y])
    plt.title(f'{x} vs {y}')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(rotation=45)
    plt.show()
#many types of plot: Strip,Violin,box,swarm
# Fill NaN values with 0 in 'Number of Medicare Beneficiaries'
df['Number of Medicare Beneficiaries'].fillna(0, inplace=True)

# Convert 'Number of Medicare Beneficiaries' to numeric and ensure no NaN values
df['Number of Medicare Beneficiaries'] = pd.to_numeric(df['Number of Medicare Beneficiaries'], errors='coerce')
df = df[df['Number of Medicare Beneficiaries'].notna()]  # Remove any rows with NaNs

# Filter out rows where 'Number of Medicare Beneficiaries' is zero or invalid
df = df[df['Number of Medicare Beneficiaries'] > 0]

# Encode gender for color mapping
label_encoder = LabelEncoder()
df['Gender Encoded'] = label_encoder.fit_transform(df['Gender of the Provider'])

# Create the static bubble plot
figure = px.scatter(df,
                    x='Average Medicare Allowed Amount',
                    y='Average Medicare Payment Amount',
                    size='Number of Medicare Beneficiaries',
                    color='Gender of the Provider',  # Color by gender
                    hover_name='National Provider Identifier',
                    title='Bubble Plot of Medicare Data',
                    size_max=50)

# Show the plot
figure.show()

if 'HCPCS Description' in data.columns:
    # Initialize the TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Fit and transform the HCPCS Description
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['HCPCS Description'])

    # Convert the TF-IDF matrix to a DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    # Concatenate the original DataFrame with the TF-IDF features
    data = pd.concat([data, tfidf_df], axis=1)

    # Create a new column showing TF-IDF as a string for each row
    data['TF-IDF Representation'] = tfidf_df.apply(
        lambda row: ', '.join([f"{col}: {row[col]:.4f}" for col in tfidf_df.columns if row[col] > 0]),
        axis=1
    )

    # Display the updated DataFrame
    print(data[['HCPCS Description', 'TF-IDF Representation']].head())
else:
    print("Column 'HCPCS Description' does not exist in the DataFrame.")


# Check if 'HCPCS Description' exists in the DataFrame
if 'HCPCS Description' in data.columns:
    # Initialize the TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Fit and transform the HCPCS Description
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['HCPCS Description'])

    # Convert the TF-IDF matrix to a DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    # Concatenate the original DataFrame with the TF-IDF features
    data = pd.concat([data, tfidf_df], axis=1)

    # Display the first few rows of the TF-IDF matrix
    print(data[['HCPCS Description'] + list(tfidf_df.columns)].head())

    # Visualize the top 20 terms in a heatmap
    top_n_terms = 20  # Number of terms to visualize
    tfidf_subset = tfidf_df.iloc[:, :top_n_terms]

    # Create a heatmap using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(tfidf_subset, cmap="Blues", annot=False, cbar=True)
    plt.title("TF-IDF Heatmap for Top 20 Terms")
    plt.xlabel("Terms")
    plt.ylabel("HCPCS Description Index")
    plt.show()

else:
    print("Column 'HCPCS Description' does not exist in the DataFrame.")


#anomaly_detection

# Data Preprocessing
# Convert columns to string before applying .str.replace
df['Number of Medicare Beneficiaries'] = pd.to_numeric(df['Number of Medicare Beneficiaries'].astype(str).str.replace(',', ''), errors='coerce')
df['Average Medicare Allowed Amount'] = pd.to_numeric(df['Average Medicare Allowed Amount'].astype(str).str.replace(',', ''), errors='coerce')
df['Average Medicare Payment Amount'] = pd.to_numeric(df['Average Medicare Payment Amount'].astype(str).str.replace(',', ''), errors='coerce')

df = df.dropna(subset=['Number of Medicare Beneficiaries', 'Average Medicare Allowed Amount', 'Average Medicare Payment Amount'])
df = df[df['Number of Medicare Beneficiaries'] > 0]

label_encoder = LabelEncoder()
df['Gender Encoded'] = label_encoder.fit_transform(df['Gender of the Provider'])

# Feature Selection and Scaling
X = df[['Average Medicare Allowed Amount', 'Number of Medicare Beneficiaries', 'Gender Encoded']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KNN Anomaly Detection
knn = NearestNeighbors(n_neighbors=5)
knn.fit(X_scaled)
distances, indices = knn.kneighbors(X_scaled)
anomaly_scores = distances.mean(axis=1)

# Set threshold for anomaly detection
threshold = np.percentile(anomaly_scores, 95)
anomalies = anomaly_scores > threshold
df['Anomaly'] = anomalies



# ADD THE NEW CODE HERE:
import pickle

# Save the trained models and necessary transformers
model_data = {
    'knn_model': knn,
    'scaler': scaler,
    'label_encoder': label_encoder,
    'threshold': threshold
}

# Save the model data to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(model_data, file)

print("Model and transformers have been saved successfully!")

# Then continue with your existing visualization code...



# Print anomalies with specific columns
anomalous_data = df[df['Anomaly']]
columns_to_display = ['Average Medicare Allowed Amount', 'Average Medicare Payment Amount', 
                      'Number of Medicare Beneficiaries', 'Gender of the Provider']

print("Anomalies detected in the dataset:")
print(anomalous_data[columns_to_display])

# Plotting anomalies in scatter plot
fig, ax = plt.subplots()
colors = ['red' if anomaly else 'blue' for anomaly in df['Anomaly']]
scat = ax.scatter(df['Average Medicare Allowed Amount'], df['Average Medicare Payment Amount'], c=colors, label='Data Points')

# Highlight anomalies
anomalous_points = anomalous_data[['Average Medicare Allowed Amount', 'Average Medicare Payment Amount']].values
ax.scatter(anomalous_points[:, 0], anomalous_points[:, 1], c='red', label='Anomalies', edgecolors='black')

plt.xlabel('Average Medicare Allowed Amount')
plt.ylabel('Average Medicare Payment Amount')
plt.title('KNN Anomaly Detection with Anomalies Highlighted')
plt.legend()
plt.show()

# Animated Scatter Plot
fig, ax = plt.subplots()
scat = ax.scatter(df['Average Medicare Allowed Amount'], df['Average Medicare Payment Amount'], c=df['Anomaly'], cmap='coolwarm')

# Function to update scatter plot
def update(frame):
    # Slightly adjust the position of the points to create movement
    df['Average Medicare Allowed Amount'] += np.random.randn(len(df)) * 0.01
    df['Average Medicare Payment Amount'] += np.random.randn(len(df)) * 0.01
    
    # Update the scatter plot with the new positions
    scat.set_offsets(np.c_[df['Average Medicare Allowed Amount'], df['Average Medicare Payment Amount']])
    
    return scat,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=100, interval=100, blit=True)

plt.xlabel('Average Medicare Allowed Amount')
plt.ylabel('Average Medicare Payment Amount')
plt.title('Animated KNN Anomaly Detection')
plt.show()


def check_anomaly():
    print("Enter the values for the following features:")
    try:
        # Get input values from the user
        allowed_amount = float(input("Average Medicare Allowed Amount: "))
        payment_amount = float(input("Average Medicare Payment Amount: "))
        num_beneficiaries = float(input("Number of Medicare Beneficiaries: "))
        gender = input("Gender of the Provider (Male/Female): ").capitalize()
        
        # Encode gender using the label encoder
        if gender not in label_encoder.classes_:
            print(f"Invalid gender! Please choose from: {list(label_encoder.classes_)}")
            return
        
        gender_encoded = label_encoder.transform([gender])[0]
        
        # Combine the inputs into a single data point
        input_data = np.array([[allowed_amount, num_beneficiaries, gender_encoded]])
        
        # Scale the input data
        input_data_scaled = scaler.transform(input_data)
        
        # Compute the anomaly score for the input data
        distances, _ = knn.kneighbors(input_data_scaled)
        anomaly_score = distances.mean(axis=1)[0]
        
        # Check if the anomaly score exceeds the threshold
        if anomaly_score > threshold:
            print("The input data point is an anomaly!")
        else:
            print("The input data point is NOT an anomaly.")
    except Exception as e:
        print(f"Error: {e}")

# Call the function to test
check_anomaly()


#NON-ANOMALOUS DATA
# Data Preprocessing
# Convert columns to string before applying .str.replace
df['Number of Medicare Beneficiaries'] = pd.to_numeric(df['Number of Medicare Beneficiaries'].astype(str).str.replace(',', ''), errors='coerce')
df['Average Medicare Allowed Amount'] = pd.to_numeric(df['Average Medicare Allowed Amount'].astype(str).str.replace(',', ''), errors='coerce')
df['Average Medicare Payment Amount'] = pd.to_numeric(df['Average Medicare Payment Amount'].astype(str).str.replace(',', ''), errors='coerce')

df = df.dropna(subset=['Number of Medicare Beneficiaries', 'Average Medicare Allowed Amount', 'Average Medicare Payment Amount'])
df = df[df['Number of Medicare Beneficiaries'] > 0]

label_encoder = LabelEncoder()
df['Gender Encoded'] = label_encoder.fit_transform(df['Gender of the Provider'])

# Feature Selection and Scaling
X = df[['Average Medicare Allowed Amount', 'Number of Medicare Beneficiaries', 'Gender Encoded']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KNN Anomaly Detection
knn = NearestNeighbors(n_neighbors=5)
knn.fit(X_scaled)
distances, indices = knn.kneighbors(X_scaled)
anomaly_scores = distances.mean(axis=1)

# Set threshold for anomaly detection
threshold = np.percentile(anomaly_scores, 95)
anomalies = anomaly_scores > threshold
df['Anomaly'] = anomalies

# Filter for non-anomalous data
non_anomalous_data = df[~df['Anomaly']]  # Select rows where 'Anomaly' is False
columns_to_display = ['Average Medicare Allowed Amount', 'Average Medicare Payment Amount', 
                      'Number of Medicare Beneficiaries', 'Gender of the Provider']

print("Non-anomalous data in the dataset:")
print(non_anomalous_data[columns_to_display])

# Plotting non-anomalous data in scatter plot
fig, ax = plt.subplots()
colors = ['blue' if not anomaly else 'red' for anomaly in df['Anomaly']]
scat = ax.scatter(df['Average Medicare Allowed Amount'], df['Average Medicare Payment Amount'], c=colors, label='Data Points')

# Highlight non-anomalous points
non_anomalous_points = non_anomalous_data[['Average Medicare Allowed Amount', 'Average Medicare Payment Amount']].values
ax.scatter(non_anomalous_points[:, 0], non_anomalous_points[:, 1], c='blue', label='Non-Anomalous', edgecolors='black')

plt.xlabel('Average Medicare Allowed Amount')
plt.ylabel('Average Medicare Payment Amount')
plt.title('KNN Anomaly Detection with Non-Anomalies Highlighted')
plt.legend()
plt.show()

# Animated Scatter Plot (Non-Anomalous)
fig, ax = plt.subplots()
scat = ax.scatter(df['Average Medicare Allowed Amount'], df['Average Medicare Payment Amount'], c=~df['Anomaly'], cmap='coolwarm')

# Function to update scatter plot
def update(frame):
    # Slightly adjust the position of the points to create movement
    df['Average Medicare Allowed Amount'] += np.random.randn(len(df)) * 0.01
    df['Average Medicare Payment Amount'] += np.random.randn(len(df)) * 0.01
    
    # Update the scatter plot with the new positions
    scat.set_offsets(np.c_[df['Average Medicare Allowed Amount'], df['Average Medicare Payment Amount']])
    
    return scat,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=100, interval=100, blit=True)

plt.xlabel('Average Medicare Allowed Amount')
plt.ylabel('Average Medicare Payment Amount')
plt.title('Animated KNN Detection (Non-Anomalous Data)')
plt.show()

def check_anomaly():
    print("Enter the values for the following features:")
    try:
        # Get input values from the user
        allowed_amount = float(input("Average Medicare Allowed Amount: "))
        payment_amount = float(input("Average Medicare Payment Amount: "))
        num_beneficiaries = float(input("Number of Medicare Beneficiaries: "))
        gender = input("Gender of the Provider (Male/Female): ").capitalize()
        
        # Encode gender using the label encoder
        if gender not in label_encoder.classes_:
            print(f"Invalid gender! Please choose from: {list(label_encoder.classes_)}")
            return
        
        gender_encoded = label_encoder.transform([gender])[0]
        
        # Combine the inputs into a single data point
        input_data = np.array([[allowed_amount, num_beneficiaries, gender_encoded]])
        
        # Scale the input data
        input_data_scaled = scaler.transform(input_data)
        
        # Compute the anomaly score for the input data
        distances, _ = knn.kneighbors(input_data_scaled)
        anomaly_score = distances.mean(axis=1)[0]
        
        # Check if the anomaly score exceeds the threshold
        if anomaly_score > threshold:
            print("The input data point is an anomaly!")
        else:
            print("The input data point is NOT an anomaly.")
    except Exception as e:
        print(f"Error: {e}")

# Call the function to test
check_anomaly()
