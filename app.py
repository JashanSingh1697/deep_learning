import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load your trained modelapp.py
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('jashanfinal_model.h5')
    return model

model = load_model()

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = image.resize((32, 32))  # Assuming your model takes 32x32 images like CIFAR-10
    img = np.array(img) / 255.0   # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Streamlit app interface
st.title("Image Label Prediction App")

# Allow user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make a prediction
    predictions = model.predict(processed_image)
    
    # Assuming you have 10 classes (like CIFAR-10), adjust accordingly
    predicted_label = np.argmax(predictions, axis=1)[0]
    st.write(f"Predicted Label: {predicted_label}")
