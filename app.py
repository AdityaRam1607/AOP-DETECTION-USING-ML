from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained models
svm_model = joblib.load("svm_model.pkl")
gb_model = joblib.load("gb_model.pkl")

# Route to render the index.html template
@app.route('/')
def index():
    return render_template('index.html')

# Function to preprocess input features
def preprocess_input(input_data):
    # Preprocess the input data as needed
    return input_data

# Function to make predictions using the hybrid model
def predict_aop(input_features):
    # Predict using SVR
    svr_pred = svm_model.predict(input_features.reshape(1, -1))

    # Augment features with SVR prediction
    augmented_features = np.append(input_features, svr_pred)

    # Predict using Gradient Boosting Regressor
    hybrid_pred = gb_model.predict(augmented_features.reshape(1, -1))

    return hybrid_pred[0]

# Route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Preprocess input data
    input_features = preprocess_input(np.array(data['input_features']))

    # Make prediction
    prediction = predict_aop(input_features)

    # Return prediction as JSON response
    return jsonify({'predicted_aop': prediction})

if __name__ == '__main__':
    app.run(debug=True)
