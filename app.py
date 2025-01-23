from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Directories and configurations
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MODEL_PATH = os.path.join(BASE_DIR, 'efficientnetv2s.h5')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels and remedies
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
    'Apple scab': "Use fungicides containing captan or myclobutanil. Practice proper pruning and dispose of fallen leaves to reduce fungal spores.",
    'Apple Black rot': "Prune infected branches and remove fallen fruit. Use fungicides such as thiophanate-methyl or captan.",
    'Apple Cedar apple rust': "Remove nearby juniper plants to break the disease cycle. Use fungicides like myclobutanil during early spring.",
    'Apple healthy': "No issues detected. Ensure regular care, proper watering, and pest management.",
    'Blueberry healthy': "No issues detected. Maintain acidic soil, water consistently, and prune for good air circulation.",
    'Cherry Powdery mildew': "Use sulfur or potassium bicarbonate fungicides. Prune affected branches and improve air circulation.",
    'Cherry healthy': "No issues detected. Regularly water and fertilize, and prune to maintain plant health.",
    'Corn Gray leaf spot': "Use resistant varieties and apply fungicides like azoxystrobin. Rotate crops to minimize disease spread.",
    'Corn Common rust': "Plant resistant varieties and use fungicides such as mancozeb or propiconazole.",
    'Corn Northern Leaf Blight': "Plant resistant hybrids and apply fungicides during early disease stages.",
    'Corn healthy': "No issues detected. Continue proper crop care and pest management.",
    'Grape Black rot': "Apply fungicides like mancozeb or myclobutanil. Remove infected fruit and leaves promptly.",
    'Grape Esca (Black Measles)': "Prune and remove infected vines. Minimize stress by providing proper irrigation and nutrients.",
    'Grape Leaf blight': "Use fungicides containing captan or mancozeb. Prune infected areas and increase spacing for airflow.",
    'Grape healthy': "No issues detected. Regularly prune and monitor for pests or diseases.",
    'Orange Citrus greening': "Control psyllid populations using insecticides. Remove infected trees to prevent spread.",
    'Peach Bacterial spot': "Use copper-based fungicides and resistant varieties. Avoid overhead watering.",
    'Peach healthy': "No issues detected. Water consistently and apply balanced fertilizers as needed.",
    'Bell Pepper Bacterial spot': "Use copper fungicides and avoid wetting leaves during watering. Remove infected plants promptly.",
    'Pepper healthy': "No issues detected. Ensure proper care and monitor for pests or diseases.",
    'Potato Early blight': "Apply fungicides like chlorothalonil or mancozeb. Rotate crops to reduce disease persistence.",
    'Potato Late blight': "Use fungicides such as mancozeb or chlorothalonil. Remove and destroy infected plants.",
    'Potato healthy': "No issues detected. Continue regular watering and pest control.",
    'Raspberry healthy': "No issues detected. Prune regularly and ensure proper soil drainage.",
    'Soybean healthy': "No issues detected. Rotate crops and manage pests effectively.",
    'Squash Powdery mildew': "Use fungicides like sulfur or neem oil. Ensure good air circulation around plants.",
    'Strawberry Leaf scorch': "Prune infected leaves and use fungicides like myclobutanil or captan.",
    'Strawberry healthy': "No issues detected. Fertilize appropriately and monitor for pests.",
    'Tomato Bacterial spot': "Apply copper fungicides and avoid overhead watering. Remove infected plants promptly.",
    'Tomato Early blight': "Use fungicides like chlorothalonil and rotate crops annually.",
    'Tomato Late blight': "Apply fungicides and ensure proper plant spacing to improve airflow.",
    'Tomato Leaf Mold': "Increase ventilation and use fungicides like mancozeb or chlorothalonil.",
    'Tomato Septoria leaf spot': "Remove infected leaves and apply fungicides early in the growing season.",
    'Tomato Spider mites': "Spray with insecticidal soap or neem oil. Increase humidity around plants.",
    'Tomato Target Spot': "Apply fungicides containing mancozeb or chlorothalonil. Remove infected leaves.",
    'Tomato Yellow Leaf Curl Virus': "Control whiteflies with insecticides or traps. Remove infected plants.",
    'Tomato mosaic virus': "Avoid handling plants when wet and use resistant varieties.",
    'Tomato healthy': "No issues detected. Provide consistent watering and fertilization."
}

# Utility function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess image for prediction
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))  # Adjust this size based on your model's input
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

@app.route('/')
def home():
    return "Flask backend is running!"
    
# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess the image
        image = preprocess_image(filepath)

        # Make prediction
        predictions = model.predict(image)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = CLASSES[predicted_class_index]
        remedy = REMEDIES.get(predicted_class, "No remedy available for this class.")

        return jsonify({'prediction': predicted_class, 'remedy': remedy})

    return jsonify({'error': 'Invalid file type'}), 400

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
