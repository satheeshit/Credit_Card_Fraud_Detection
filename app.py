from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load trained model
model_path = "D:\model.pkl"  # Use a relative path
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Fit scaler on the same feature set used in training
scaler = StandardScaler()
X_train_sample = np.array([[0, 0, 0, 0, 0]])  # Replace with actual training mean/std
scaler.fit(X_train_sample)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure feature order matches training
        feature_names = ['V14', 'V10', 'V4', 'V11', 'V12']
        features = [float(request.form[f]) for f in feature_names]
        
        # Transform input features
        scaled_features = scaler.transform([features])
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        result = "Fraudulent Transaction" if prediction == 1 else "Legitimate Transaction"
        
        return render_template("result.html", prediction=result)
    
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
