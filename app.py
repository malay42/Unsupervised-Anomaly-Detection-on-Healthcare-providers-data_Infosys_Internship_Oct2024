from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go
import json

app = Flask(__name__)

# Load the saved model and transformers
def load_model():
    with open('model.pkl', 'rb') as file:
        model_data = pickle.load(file)
    return model_data

try:
    model_data = load_model()
    knn = model_data['knn_model']
    scaler = model_data['scaler']
    label_encoder = model_data['label_encoder']
    threshold = model_data['threshold']
except Exception as e:
    print(f"Error loading model: {str(e)}")

def create_plot(anomaly_score, threshold):
    # Create a gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = anomaly_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Anomaly Score"},
        gauge = {
            'axis': {'range': [None, max(anomaly_score * 1.2, threshold * 1.2)]},
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold
            },
            'steps': [
                {'range': [0, threshold], 'color': "lightgreen"},
                {'range': [threshold, max(anomaly_score * 1.2, threshold * 1.2)], 'color': "lightpink"}
            ]
        }
    ))
    
    # Update layout
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return json.loads(fig.to_json())

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the form
        allowed_amount = float(request.form['allowed_amount'])
        num_beneficiaries = int(request.form['num_beneficiaries'])
        gender = request.form['gender']
        
        # Encode gender
        gender_encoded = label_encoder.transform([gender])[0]
        
        # Prepare input data
        input_data = np.array([[allowed_amount, num_beneficiaries, gender_encoded]])
        
        # Scale the input data
        input_data_scaled = scaler.transform(input_data)
        
        # Compute the anomaly score
        distances, _ = knn.kneighbors(input_data_scaled)
        anomaly_score = distances.mean(axis=1)[0]
        
        # Create plot data
        plot_data = create_plot(anomaly_score, threshold)
        
        # Prepare result
        result = {
            'score': float(anomaly_score),
            'threshold': float(threshold),
            'is_anomaly': bool(anomaly_score > threshold)
        }
        
        return render_template('index.html', prediction=result, plot_data=plot_data)
        
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)