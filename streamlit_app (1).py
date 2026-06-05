import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Image Classification", layout="centered")

@st.cache_resource
def load_model():
    model_path = "load_image_classification_model.h5"
    if not os.path.exists(model_path):
        st.error(f"Model tidak ditemukan: {model_path}")
        return None
    return tf.keras.models.load_model(model_path)

model = load_model()

st.title("🖼️ Image Classification")
st.write("Upload gambar untuk melakukan klasifikasi Positive / Negative")

uploaded_file = st.file_uploader(
    "Pilih gambar",
    type=["jpg", "jpeg", "png"]
)

target_size = (128, 128)
class_labels = {0: "negative", 1: "positive"}

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Gambar yang diunggah", use_container_width=True)

    img = image.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    probability = float(prediction[0][0])

    if probability > 0.5:
        predicted_class = 1
    else:
        predicted_class = 0

    label = class_labels[predicted_class]

    st.subheader("Hasil Prediksi")
    st.success(f"Kelas: {label}")
    st.info(f"Probabilitas: {probability * 100:.2f}%")

    st.write("Raw Prediction:", prediction)
