import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the SVM model from the pickle file
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

def predict(values):
    # Convert the input values to a numpy array and reshape it to a single row
    X = np.array(values).reshape(1, -1)
    
    # Use the SVM model to make predictions on the input values
    y_pred = model.predict(X)
    
    # Return the predicted result as a string
    if y_pred[0] == 1:
        return 'Churn'
    else:
        return 'No churn'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_result():
    # Get the input values from the form
    gender = int(request.form['gender'])
    tenure = int(request.form['tenure'])
    phone_service = int(request.form['phone_service'])
    internet_service = int(request.form['internet_service'])
    online_security = int(request.form['online_security'])
    online_backup = int(request.form['online_backup'])
    tech_support = int(request.form['tech_support'])
    contract = int(request.form['contract'])
    paperless_billing = int(request.form['paperless_billing'])
    payment_method = int(request.form['payment_method'])
    total_charges = float(request.form['total_charges'])

    # Pass the input values to the predict() function to get the predicted result
    result = model.predict([[gender, tenure, phone_service, internet_service, online_security, online_backup, tech_support, contract, paperless_billing, payment_method, total_charges]])

    # Render the result.html template with the predicted result
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
