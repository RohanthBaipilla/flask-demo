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
        # Load image and preprocess for both models
        image = Image.open(io.BytesIO(file.read()))
        
        # Preprocess image for CNN
        cnn_image = preprocess_image_for_cnn(image)
        cnn_prediction = cnn_model.predict(cnn_image)
        
        # Preprocess image for ResNet
        resnet_image = preprocess_image_for_resnet(image)
        resnet_prediction = resnet_model.predict(resnet_image)

        # Determine predicted classes
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
    if os.path.exists(cnn_model_path) and os.path.exists(resnet_model_path):
        return jsonify({'status': 'Both model files found'}), 200
    else:
        return jsonify({'status': 'One or both model files not found'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

 # Bind to all interfaces and set port
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# from PIL import Image
# import numpy as np
# import io

# app = Flask(__name__)
# CORS(app)

# model_path = 'breast_cancer_cnn_model.h5'

# try:
#     model = load_model(model_path)
# except Exception as e:
#     print(f"Error loading model: {e}")
#     exit(1)

# def preprocess_image(image):
#     image = image.convert('L')
#     image = image.resize((128, 128))
#     image = img_to_array(image)
#     image = np.expand_dims(image, axis=0)
#     image = image / 255.0
#     return image

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file provided'}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No file selected'}), 400

#     try:
#         image = Image.open(io.BytesIO(file.read()))
#         image = preprocess_image(image)
#         prediction = model.predict(image)
#         predicted_class = 'Cancer (Malignant)' if prediction[0][0] >= 0.5 else 'Not Cancer (Benign)'
#         return jsonify({
#             'prediction_probability': float(prediction[0][0]),
#             'predicted_class': predicted_class
#         })
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/test-model', methods=['GET'])
# def test_model():
#     if os.path.exists(model_path):
#         return jsonify({'status': 'Model file found'}), 200
#     else:
#         return jsonify({'status': 'Model file not found'}), 404
