# app.py
import os
import numpy as np
from flask import Flask, request, render_template
from flask_cors import CORS  # Import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model = load_model('sickle_cell_detection_model.h5')

# Define labels for predictions
labels = {0: 'Negative', 1: 'Positive'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return {"error": "No file uploaded"}, 400

    file = request.files['file']
    if file.filename == '':
        return {"error": "No file uploaded"}, 400

    # Save the uploaded image
    img_path = os.path.join('static', file.filename)
    file.save(img_path)

    # Preprocess the image
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = labels[np.argmax(predictions)]
    probabilities = {labels[i]: round(prob * 100, 2) for i, prob in enumerate(predictions[0])}

    return {"prediction": predicted_class, "probabilities": probabilities, "image_path": img_path}

if __name__ == '__main__':
    app.run(debug=True)
