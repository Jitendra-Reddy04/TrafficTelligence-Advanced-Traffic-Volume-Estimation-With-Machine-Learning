# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python [conda env:base] *
#     language: python
#     name: conda-base-py
# ---

# %%
import numpy as np
import pickle
from flask import Flask, request, render_template

# Initialize Flask app
app = Flask(__name__)

# Load trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scale = pickle.load(open('scale.pkl', 'rb'))

@app.route('/')  # Home page route
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])  # Prediction route
def predict():
    try:
        # Extracting 11 input features from the form (in correct order)
        features = [
            float(request.form['holiday']),
            float(request.form['temp']),
            float(request.form['rain']),
            float(request.form['snow']),
            float(request.form['weather']),
            float(request.form['day']),
            float(request.form['month']),
            float(request.form['year']),
            float(request.form['hours']),
            float(request.form['minutes']),
            float(request.form['seconds'])
        ]

        # Convert to numpy array and reshape for prediction
        final_input = np.array(features).reshape(1, -1)

        # Scale the input
        final_input_scaled = scale.transform(final_input)

        # Predict traffic volume
        prediction = model.predict(final_input_scaled)

        # Return result to frontend
        return render_template('index.html', prediction_text=f'Predicted Traffic Volume: {prediction[0]:.0f}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'❌ Error: {e}')

# Run the app
if __name__ == '_main_':
    app.run(debug=True)

# %%


# %%
# import os
# os.getcwd()

# %%
