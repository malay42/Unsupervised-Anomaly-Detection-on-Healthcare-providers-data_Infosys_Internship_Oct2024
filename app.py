from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
iforest_pipeline = joblib.load('iforest_pipeline.pkl')
power_transformer = iforest_pipeline['power_transformer']
iforest_model = iforest_pipeline['iforest_model']
autoencoder = load_model('autoencoder_model.keras')

def preprocess_input(input_data):
    input_df = pd.DataFrame(input_data, columns=[
        'Number of Services',
        'Number of Medicare Beneficiaries',
        'Number of Distinct Medicare Beneficiary/Per Day Services',
        'Average Medicare Allowed Amount',
        'Average Submitted Charge Amount',
        'Average Medicare Payment Amount',
        'Average Medicare Standardized Amount',
        'Gender of the Provider'
    ])
    for col in input_df.columns:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
    input_df = input_df.dropna()
    
    # Apply PowerTransformer
    input_scaled = power_transformer.transform(input_df)
    
    return input_scaled

def predict(input_data):
    input_scaled = preprocess_input(input_data)
    iforest_predictions = iforest_model.predict(input_scaled)
    autoencoder_predictions = autoencoder.predict(input_scaled)
    reconstruction_errors = np.mean(np.square(input_scaled - autoencoder_predictions), axis=1)
    
    return iforest_predictions, reconstruction_errors

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    if request.method == 'POST':
        
        input_data = [
            [
                float(request.form['num_services']),
                float(request.form['num_beneficiaries']),
                float(request.form['num_distinct_services']),
                float(request.form['avg_allowed_amount']),
                float(request.form['avg_submitted_charge']),
                float(request.form['avg_payment']),
                float(request.form['avg_standardized']),
                int(request.form['gender'])
            ]
        ]
        iforest_preds, reconstruction_errors = predict(input_data)    
        results = []
        for i, (iforest_pred, error) in enumerate(zip(iforest_preds, reconstruction_errors)):
            if iforest_pred == 1:
                results.append(f"Normal: Reconstruction error = {error:.4f}")
            else:
                results.append(f"Anomaly Detected: Reconstruction error = {error:.4f}")

    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
