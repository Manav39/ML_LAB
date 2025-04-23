import streamlit as st
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import uuid

dataset_info = tfds.builder('food101').info
labels = dataset_info.features['label'].names

model = tf.keras.applications.MobileNetV2(weights='imagenet')

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def detect_food_items(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        input_tensor = preprocess_image(img)

        predictions = model.predict(input_tensor)
        decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions)[0]

        food_items = [label for (_, label, _) in decoded]

        return food_items
    except Exception as e:
        return [str(e)]

def extract_ingredients_from_image(image_path):
    return detect_food_items(image_path)

st.title("Food Items Extractor 🍽️")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Extract Ingredients"):
        temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
        image.save(temp_filename)

        ingredients = extract_ingredients_from_image(temp_filename)

        os.remove(temp_filename)

        st.subheader("Identified Ingredients:")
        for item in ingredients:
            st.write(f"• {item}")