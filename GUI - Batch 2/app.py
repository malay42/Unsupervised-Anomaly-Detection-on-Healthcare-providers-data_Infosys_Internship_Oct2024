from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.api.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load the saved model and scaler
model_path = "autoencoder_model.h5"
scaler_path = "scaler_data.csv"

autoencoder = load_model(model_path)
scaler = StandardScaler()
scaler_data = pd.read_csv(scaler_path)
scaler.fit(scaler_data.values)  # Fit the scaler with saved scaled data

# Numeric columns expected
numeric_columns = [
    'Number of Services', 'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services',
    'Average Medicare Allowed Amount', 'Average Submitted Charge Amount',
    'Average Medicare Payment Amount', 'Average Medicare Standardized Amount'
]

# Load the best threshold (set a placeholder or use a saved value)
best_threshold = 20  # Replace with the actual threshold found during training

def detect_anomaly_in_row(row, scaler, autoencoder, best_threshold):
    row_data = np.array([row]).reshape(1, -1)
    row_data = scaler.transform(row_data)
    row_pred = autoencoder.predict(row_data)
    row_error = np.mean(np.square(row_data - row_pred))
    is_anomaly = row_error > best_threshold
    return is_anomaly, row_error

# Route for the form page
@app.route("/")
def index():
    return render_template("index.html", numeric_columns=numeric_columns)

# Route to handle form submission
@app.route("/detect", methods=["POST"])
def detect_anomaly():
    try:
        # Collect data from the form
        row = [float(request.form[column]) for column in numeric_columns]

        # Detect anomaly
        is_anomaly, row_error = detect_anomaly_in_row(row, scaler, autoencoder, best_threshold)

        # Prepare the result to display
        result = "Anomaly Detected!" if is_anomaly else "Normal Data Point"
        return render_template("result.html", result=result, error=row_error)

    except Exception as e:
        return render_template("result.html", result="Error", error=str(e))

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
