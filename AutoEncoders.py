import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.losses import MeanSquaredLogarithmicError
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("Cleaned_Healthcare Providers.csv")
columns = ["Number of Services", 
           "Number of Medicare Beneficiaries",
           "Number of Distinct Medicare Beneficiary/Per Day Services",
           "Average Medicare Allowed Amount",
           "Average Submitted Charge Amount",
           "Average Medicare Payment Amount",
           "Average Medicare Standardized Amount"]
df = df[columns]

# Data scaling using StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(df)

# Train-test split for the model (since it's unsupervised, we just use it for validation later)
x_train, x_test = train_test_split(x_scaled, test_size=0.2, random_state=42)

# Define Autoencoder Model
class AutoEncoder(Model):
    def __init__(self, output_units, code_size=8):
        super().__init__()
        self.encoder = Sequential([
            Dense(256, activation='relu'),
            Dropout(0.1),
            Dense(128, activation='relu'),
            Dropout(0.1),
            Dense(64, activation='relu'),
            Dropout(0.1),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(code_size, activation='relu')
        ])
        self.decoder = Sequential([
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(64, activation='relu'),
            Dropout(0.1),
            Dense(128, activation='relu'),
            Dropout(0.1),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(output_units, activation='sigmoid')
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

# Create and compile the autoencoder model
model = AutoEncoder(output_units=x_train.shape[1])
model.compile(loss=MeanSquaredLogarithmicError(), optimizer=Adam())

# Train the model
history = model.fit(x_train, x_train, epochs=20, batch_size=32, validation_data=(x_test, x_test))
