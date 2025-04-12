from flask import Flask, request, jsonify
import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
import easyocr
import numpy as np
import google.generativeai as genai
import PIL
from flask_cors import CORS
from PIL import Image
import os

genai.configure(api_key="AIzaSyCHG8tsYSlO9EEEHJhUbUTDQB0tOkclqm8")

app = Flask(__name__)
CORS(app)
reader = easyocr.Reader(['en'], gpu=False)

# Load dataset info to get labels
dataset_info = tfds.builder('food101').info
labels = dataset_info.features['label'].names

# Load pretrained model (MobileNetV2 + Food101)
model = tf.keras.applications.MobileNetV2(weights='imagenet')

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

@app.route('/food-detect', methods=['POST'])
def detect_food_items():
    try:
        image_file = request.files['image']
        image_path = "temp_image.jpg"
        image_file.save(image_path)

        img = Image.open(image_path).convert("RGB")
        input_tensor = preprocess_image(img)

        # Predict using ImageNet labels (not food-specific but includes many foods)
        predictions = model.predict(input_tensor)
        decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions)[0]

        # Extract top predictions
        food_items = [label for (_, label, _) in decoded]

        os.remove(image_path)
        print(food_items)
        return jsonify({'result': food_items})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ingredientsfetch', methods=['POST'])
def ingredientsfetch():
    try:
        image_file = request.files['image']
        image_file.save("temp_image.jpg")
        img = PIL.Image.open('temp_image.jpg')
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = "Extract all the food ingredients from the following text without any special characters or numbers just , as the separator between ingredients"
        result = model.generate_content([prompt,img],stream=True)
        result.resolve()
        os.remove("temp_image.jpg")
        return jsonify({'result': result.text})
    except Exception as e:
        print(str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/geminiocr', methods=['POST'])
def geminiocr():
    try:
        image_file = request.files['image']
        image_file.save("temp_image.jpg")
        img = PIL.Image.open('temp_image.jpg')
        model = genai.GenerativeModel('gemini-1.5-flash')
        result = model.generate_content([img,"Extract all the food ingredients from the image without any special characters or numbers just , as the separator between ingredients"],stream=True)
        result.resolve()
        os.remove("temp_image.jpg")
        return jsonify({'result': result.text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)
