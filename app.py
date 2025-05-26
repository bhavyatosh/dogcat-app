import streamlit as st
import gdown
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# File details
model_path = 'dog_cat_model.h5'
gdrive_file_id = '1a79bIzeAot2kqt2y8VFNJfVe-Qs8Y6ZJ'
gdown_url = f'https://drive.google.com/uc?id={gdrive_file_id}'


# Download the model if not already present
if not os.path.exists(model_path):
    st.write("Downloading model from Google Drive...")
    gdown.download(gdown_url, model_path, quiet=False)

# Load the trained model
model = load_model(model_path)

# App title
st.title("🐾 Dog vs Cat Classifier")
st.write("Upload an image of a dog or a cat and the model will predict which one it is.")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image for prediction
    img = img.resize((150, 150))  # Match input shape
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize

    # Make prediction
    prediction = model.predict(img_array)[0][0]
    label = "Dog 🐶" if prediction > 0.5 else "Cat 🐱"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    # Show prediction result
    st.write(f"### Prediction: {label}")
    st.write(f"Confidence: {confidence:.2%}")
