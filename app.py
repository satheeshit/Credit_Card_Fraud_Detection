from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging
import os

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Paths
model_path = r"D:\model_sgd.pkl"
scaler_path = r"D:\scaler.pkl"

# Important feature columns
feature_names = ['V14', 'V10', 'V4', 'V11', 'V12']

# Load model and scaler
def load_model_and_scaler():
    global model, scaler
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        logging.info("Model loaded successfully.")
    else:
        raise FileNotFoundError("Model file not found!")

    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        logging.info("Scaler loaded successfully.")
    else:
        raise FileNotFoundError("Scaler file not found!")

load_model_and_scaler()

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features safely
        features = [float(request.form[f]) for f in feature_names]

        # Handle optional threshold input
        threshold_str = request.form.get('threshold')
        threshold = float(threshold_str) if threshold_str else 0.2

        # Scale input
        scaled_features = scaler.transform([features])

        # Predict
        probabilities = model.predict_proba(scaled_features)[0]
        fraud_prob = probabilities[1]
        logging.debug(f"Fraud probability: {fraud_prob}")

        # Result
        result = "Fraudulent Transaction" if fraud_prob > threshold else "Legitimate Transaction"

        return render_template("result.html", prediction=result, probability=fraud_prob)

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return f"Error: {str(e)}"

@app.route('/update', methods=['POST'])
def update_model():
    global model
    try:
        file = request.files["file"]
        new_data = pd.read_csv(file)

        # Select and clean only required columns
        new_data = new_data[feature_names + ['Class']]

        # Convert all to numeric (invalid entries -> NaN)
        for col in new_data.columns:
            new_data[col] = pd.to_numeric(new_data[col], errors='coerce')

        # Drop rows with any NaN
        new_data.dropna(inplace=True)

        # Extract features and target
        X_new = new_data[feature_names]
        y_new = new_data['Class']
        X_new_scaled = scaler.transform(X_new)

        # Update the model
        model.partial_fit(X_new_scaled, y_new)

        # Save and reload model
        joblib.dump(model, model_path)
        model = joblib.load(model_path)

        return "Model updated successfully!"

    except Exception as e:
        logging.error(f"Update error: {str(e)}")
        return f"Error during update: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
