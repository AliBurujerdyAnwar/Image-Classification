import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Image Classification", page_icon="🖼️")

st.title("🖼️ Image Classification")
st.write("Upload gambar untuk mendeteksi Positive atau Negative")

MODEL_PATH = "load_image_classification_model.h5"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

TARGET_SIZE = (128, 128)

def predict_image(img):
    img = img.resize(TARGET_SIZE)
    img_array = np.array(img)

    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)

    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    pred = model.predict(img_array)
    prob = float(pred[0][0])

    label = "Positive" if prob > 0.5 else "Negative"
    return label, prob

uploaded_file = st.file_uploader(
    "Upload gambar",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diupload", use_container_width=True)

    if st.button("Prediksi"):
        label, prob = predict_image(image)

        st.subheader("Hasil Prediksi")
        st.write(f"Kelas: **{label}**")
        st.write(f"Probabilitas: **{prob*100:.2f}%**")
