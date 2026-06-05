import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(
    page_title="Image Classification",
    page_icon="🖼️",
    layout="wide"
)

MODEL_FILE = "load_image_classification_model.h5"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_FILE)

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

target_size = (128, 128)
class_labels = {0: "negative", 1: "positive"}

def predict_image(img):
    img = img.convert("RGB")
    img = img.resize(target_size)

    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)
    probability = float(prediction[0][0])

    if probability > 0.5:
        predicted_class = 1
        confidence = probability
    else:
        predicted_class = 0
        confidence = 1 - probability

    return class_labels[predicted_class], confidence, probability

st.title("🔍 Concrete Crack Image Classification")

st.write(
    "Unggah gambar untuk melakukan klasifikasi menggunakan model "
    "`load_image_classification_model.h5`."
)

uploaded_file = st.file_uploader(
    "Pilih gambar",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Gambar Input", use_container_width=True)

    with col2:
        label, confidence, probability = predict_image(image)

        st.subheader("Hasil Prediksi")

        if label.lower() == "positive":
            st.success(f"Positive")
        else:
            st.info(f"Negative")

        st.metric("Confidence", f"{confidence*100:.2f}%")
        st.write(f"Raw Probability: {probability:.4f}")

        st.progress(min(max(probability, 0.0), 1.0))

st.markdown("---")
st.caption("Dikonversi dari notebook Jupyter ke Streamlit")
