# Healthcare Anomaly Detection System

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model Details](#model-details)
- [Web Application](#web-application)
- [Results](#results)
- [Future Scope](#future-scope)

## Introduction
Healthcare fraud is a growing concern, leading to billions of dollars in losses annually. This project implements an **unsupervised anomaly detection system** to identify fraudulent activities in healthcare provider data. The system combines machine learning techniques like **Isolation Forest** and **Autoencoders** to detect anomalies efficiently.

## Features
- Real-time anomaly detection for healthcare provider data.
- Integration of multiple algorithms: Isolation Forest, One-Class SVM, and Autoencoders.
- User-friendly web interface with dark and light modes.
- Input data normalization and preprocessing for accurate results.
- Interactive visualizations for insights.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/malay42/Unsupervised-Anomaly-Detection-on-Healthcare-providers-data_Infosys_Internship_Oct2024.git
2. Navigate to the project directory:
   ```bash
   cd Unsupervised-Anomaly-Detection-on-Healthcare-providers-data_Infosys_Internship_Oct2024
3. Set up a virtual environment:
   ```bash
   python -m venv myenv
   source myenv/bin/activate   # For Linux/MacOS
   myenv\Scripts\activate      # For Windows

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
5. Install Git LFS for large model files:
   ```bash
   git lfs install
   git lfs pull

## Usage
### Web Application
1. Start the Flask server:
   ```bash
   python app.py
2. Open the web application in your browser:
   ```bash
   http://127.0.0.1:5000
3. Enter the healthcare data values manually to detect anomalies.

## Data
The dataset contains information about healthcare providers, services, and financial details. Key features include:

* Numerical Features: Number of Services, Medicare Beneficiaries, Average Charge Amounts, etc.
* Categorical Features: Gender of the Provider, Entity Type.
  
Preprocessing steps:
* Missing values handling.
* Transformation of numerical data using L2 Normalization and Power Transformer.
* Separation of numerical and categorical data for effective modeling.
  
## Model Details

### 1. Isolation Forest:
    Efficient for high-dimensional data.
    Tuned using GridSearchCV for maximum accuracy.
    
### 2. Autoencoders:
    Reconstruct input data and detect anomalies based on reconstruction error.
    Threshold for anomaly detection determined by reconstruction error distribution.
    
## Web Application

### The web application provides:
    Manual entry for data input.
    Real-time predictions (Normal/Anomaly).
    Interactive interface with dark and light mode.

## Screenshots

<img width="645" alt="image" src="https://github.com/user-attachments/assets/70bd047c-26dc-4a8a-aae0-c3e5682163da">
<img width="645" alt="image" src="https://github.com/user-attachments/assets/e6e304de-200f-40b5-a699-6f41b5f95b3f">



## Results
* Accuracy: 92.7%
* Anomalies Detected: 10,215
* False Positives: Reduced by 8.4%
* Execution Time: ~5.8 seconds for combined models.

  
## Future Scope
* Scalability: Extend the system for larger datasets using distributed processing.
* Advanced Models: Incorporate deep learning techniques for enhanced anomaly detection.
* Real-Time Monitoring: Develop pipelines for continuous fraud detection.
* Feature Expansion: Include domain-specific features for richer analysis.
* User Interface Enhancements: Add visualizations and support for multi-language accessibility.


### Author: Akashdip Saha
### Project Guide: Sir Malay
### Acknowledgment: Infosys Springboard Internship 5.0


