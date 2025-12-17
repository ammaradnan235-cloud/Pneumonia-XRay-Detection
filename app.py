# app2.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Pneumonia Detection System",
    page_icon="🫁",
    layout="centered"
)

st.title("🫁 Pneumonia Detection System (AI Powered)")
st.write("Upload a chest X-ray image to detect Pneumonia or Normal lungs.")

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("pneumonia_model.keras")
    return model

model = load_model()

# ---------------------------
# Image Preprocessing
# ---------------------------
def preprocess_image(img):
    """
    Preprocess the uploaded image to match the model input.
    """
    img = img.convert("RGB")           # Ensure 3 channels
    img = img.resize((128, 128))       # Resize to model input
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    return img_array

# ---------------------------
# File Uploader
# ---------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Open and display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Chest X-ray", use_column_width=True)

    if st.button("🔍 Detect Pneumonia"):
        with st.spinner("Analyzing X-ray..."):
            # Preprocess and predict
            img_array = preprocess_image(img)
            prediction = model.predict(img_array)
            prob = float(prediction[0][0])

            # Display result
            if prob > 0.5:
                st.error(f"🟥 Pneumonia Detected ({prob*100:.2f}%)")
            else:
                st.success(f"🟩 Normal Lungs ({(1-prob)*100:.2f}%)")

