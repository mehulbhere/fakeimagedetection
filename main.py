import streamlit as st
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import io
import contextlib

# Global variables for model information
model_names = ['CNN', 'CNN_180k', 'VGG16','VGG16_180k']
models = {name: load_model(name+'.h5') for name in model_names}

# Function to preprocess the image
def preprocess_image(image, target_size):
    st.write("Preprocessing image...")
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize pixel values
    st.write("Image preprocessed.")
    return image

# Function to perform prediction using the selected model
def predict_fake_image(image, model_name):
    st.write("Performing prediction...")
    model = models[model_name]
    prediction = model.predict(np.expand_dims(image, axis=0))
    st.write("Prediction completed.")
    return prediction[0][0]

# Function to handle image selection and display results
def select_image():
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_info = {
            "File Name": uploaded_file.name,
            "Size": f"{uploaded_file.size} bytes",
            "Resolution": f"{image.width}x{image.height}",
            "Format": image.format
        }
        st.write("Image Metadata:")
        st.write(image_info)
        st.image(image, caption='Uploaded Image', use_column_width=200)
        image = np.array(image)
        image = preprocess_image(image, target_size=(256, 256))  # Resize to (256, 256) to match model's input size
        selected_model = st.sidebar.selectbox("Select Model", model_names)
        if st.sidebar.button("Predict"):
            prediction = predict_fake_image(image, selected_model)
            st.code(f"Prediction: {'Fake'}({prediction*100:.4f}%)")
            st.write("Model Information:")
            with io.StringIO() as model_info_buffer, contextlib.redirect_stdout(model_info_buffer):
                models[selected_model].summary()
                model_info_str = model_info_buffer.getvalue()
            st.code(model_info_str)

# Streamlit app setup
st.title("Fake Image Detection")
select_image()
