Credit Card Fraud Detection - Flask Web App
A machine learning web application that detects fraudulent credit card transactions based on key features using logistic regression or SGD classifier. Built using Flask, with a modern HTML frontend and a live update feature to continuously train the model with new data.

 Features
✅ Predict if a transaction is fraudulent or legitimate using 5 important features:

V14, V10, V4, V11, V12

✅ Scaled inputs for consistent prediction accuracy.

✅ Web form to input transaction data.

✅ Optional threshold slider to control fraud sensitivity.

✅ Upload CSV file to update and incrementally train the model (online learning).

✅ Stylish and responsive UI using custom HTML and CSS

How It Works
User inputs values for 5 selected features.

Data is scaled using a pre-trained StandardScaler.

The model predicts the probability of fraud.

If the probability exceeds the set threshold (default: 0.2), the transaction is labeled fraudulent.

Users can upload a .csv file with new data to update the model on the fly using partial_fit()

Sample Inputs to Test
Use this sample for a fraudulent case:
V14: -2.3122265423263002
V10: -2.3122265423263002
V4:   2.3122265423263002
V11:  2.3122265423263002
V12: -2.3122265423263002

 Installation & Running
 # 1. Clone this repository
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# 2. Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install required packages
pip install -r requirements.txt

# 4. Run the app
python app.py

# Visit http://127.0.0.1:5000 in your browser

Model Training (Notebook)
Model training is handled in Credit_Card_Fraud_Detection.ipynb, where:

Important features are selected based on XGBoost importance.

The model is trained using an SGD Classifier with partial_fit() support.

Both the model and scaler are saved using joblib.

 CSV Upload Format
Uploaded .csv files should have these 6 columns:
V14, V10, V4, V11, V12, Class

Requirements
Flask
numpy
pandas
scikit-learn
joblib
(Use pip install -r requirements.txt)

 Author
Name: Satheesh

Email: rsatheeshit@gmail.com

GitHub: satheeshit

