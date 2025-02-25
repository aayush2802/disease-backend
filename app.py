from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS for cross-origin requests
import os
import tensorflow as tf
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Directories and configurations
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'efficientnetv2s.h5')

# Load the model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Class labels
CLASSES = [
    'Apple scab', 'Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy', 'Blueberry healthy',
    'Cherry Powdery mildew', 'Cherry healthy', 'Corn Gray leaf spot', 'Corn Common rust',
    'Corn Northern Leaf Blight', 'Corn healthy', 'Grape Black rot', 'Grape Esca (Black Measles)',
    'Grape Leaf blight', 'Grape healthy', 'Orange Citrus greening', 'Peach Bacterial spot',
    'Peach healthy', 'Bell Pepper Bacterial spot', 'Pepper healthy', 'Potato Early blight',
    'Potato Late blight', 'Potato healthy', 'Raspberry healthy', 'Soybean healthy',
    'Squash Powdery mildew', 'Strawberry Leaf scorch', 'Strawberry healthy',
    'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold',
    'Tomato Septoria leaf spot', 'Tomato Spider mites', 'Tomato Target Spot',
    'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus', 'Tomato healthy'
]

# Remedies for each class
REMEDIES = {
    'Apple scab': "Use fungicides containing captan or myclobutanil...",
    'Apple Black rot': "Prune infected branches and remove fallen fruit...",
    'Apple Cedar apple rust': "Remove nearby juniper plants...",
    'Tomato healthy': "No issues detected. Provide consistent watering...",
}

# Preprocess Base64 image for prediction
def preprocess_image(image_base64):
    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data)).convert('RGB')
        image = image.resize((224, 224))  # Adjust based on model input size
        image_array = np.array(image) / 255.0  # Normalize image
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        return image_array
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return None

@app.route('/')
def home():
    return jsonify({'message': "Flask backend is running!"})

# Prediction endpoint (now accepts Base64 images)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        # Preprocess Base64 image
        image = preprocess_image(data['image'])
        if image is None:
            return jsonify({'error': 'Error processing image'}), 500

        # Make prediction
        predictions = model.predict(image)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = CLASSES[predicted_class_index]
        remedy = REMEDIES.get(predicted_class, "No remedy available for this class.")

        print(f"✅ Prediction: {predicted_class}")
        return jsonify({'prediction': predicted_class, 'remedy': remedy})

    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
