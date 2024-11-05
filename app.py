# app.py

import os
import pickle
import numpy as np
from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.models import model_from_json

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the trained model
model_path = '36466_model.pkl'
with open(model_path, 'rb') as file:
    model_data = pickle.load(file)

# Assuming model_data contains the model structure and weights
model = model_from_json(model_data['model_structure'])
model.set_weights(model_data['model_weights'])

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the form and convert 'Yes'/'No' to binary values
    int_features = [1 if x == 'yes' else 0 for x in request.form.values()]
    final_features = np.array(int_features).reshape(1, -1)  # Reshape for model input

    # Make prediction
    prediction = model.predict(final_features)
    output = 'Postpartum Depression Detected' if prediction[0][0] >= 0.5 else 'No Postpartum Depression Detected'

    # Render the result on the same page
    return render_template('index.html', prediction_text=output)

if __name__ == "__main__":
    app.run(debug=True)
