from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io
import os

app = Flask(__name__)
CORS(app)

# Directly set model paths
cnn_model_path = 'breast_cancer_cnn_model.h5'
resnet_model_path = 'breast_cancer_resnet_model.h5'

# Load both models at startup
try:
    cnn_model = load_model(cnn_model_path)
    resnet_model = load_model(resnet_model_path)
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

def preprocess_image_for_cnn(image):
    """Preprocess the image for CNN model (grayscale)."""
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((128, 128))  # Resize to the target size
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize image
    return image

def preprocess_image_for_resnet(image):
    """Preprocess the image for ResNet model (RGB)."""
    image = image.convert('RGB')  # Convert to RGB
    image = image.resize((128, 128))  # Resize to the target size
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize image
    return image

@app.route('/predict-both', methods=['POST'])
def predict_both():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        image = Image.open(io.BytesIO(file.read()))
        
        # CNN prediction
        cnn_image = preprocess_image_for_cnn(image)
        cnn_prediction = cnn_model.predict(cnn_image)
        
        # ResNet prediction
        resnet_image = preprocess_image_for_resnet(image)
        resnet_prediction = resnet_model.predict(resnet_image)

        cnn_predicted_class = 'Cancer (Malignant)' if cnn_prediction[0][0] >= 0.5 else 'Not Cancer (Benign)'
        resnet_predicted_class = 'Cancer (Malignant)' if resnet_prediction[0][0] >= 0.5 else 'Not Cancer (Benign)'

        return jsonify({
            'cnn_predicted_class': cnn_predicted_class,
            'cnn_prediction_probability': float(cnn_prediction[0][0]),
            'resnet_predicted_class': resnet_predicted_class,
            'resnet_prediction_probability': float(resnet_prediction[0][0])
        })
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/test-model', methods=['GET'])
def test_model():
    try:
        # Simple test to verify models are working
        dummy_image = np.zeros((1, 128, 128, 1))  # Grayscale for CNN
        dummy_rgb_image = np.zeros((1, 128, 128, 3))  # RGB for ResNet

        cnn_output = cnn_model.predict(dummy_image)
        resnet_output = resnet_model.predict(dummy_rgb_image)

        return jsonify({
            'cnn_test_output': float(cnn_output[0][0]),
            'resnet_test_output': float(resnet_output[0][0]),
            'message': 'Models are loaded and responding'
        })
    except Exception as e:
        print(f"Error in test-model route: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
