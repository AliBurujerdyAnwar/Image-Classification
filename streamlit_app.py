import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ======================================================
# KONFIGURASI HALAMAN
# ======================================================
st.set_page_config(
    page_title="Image Classification",
    page_icon="🖼️",
    layout="centered"
)

st.title("🖼️ Image Classification")
st.write("Upload gambar untuk mendeteksi apakah gambar termasuk **Positive** atau **Negative**.")

# ======================================================
# LOAD MODEL
# ======================================================
MODEL_PATH = "load_image_classification_model.h5"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
    st.success("Model berhasil dimuat.")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# ======================================================
# PARAMETER
# ======================================================
TARGET_SIZE = (128, 128)
CLASS_LABELS = {
    0: "Negative",
    1: "Positive"
}

# ======================================================
# FUNGSI PREDIKSI
# ======================================================
def predict_image(img, model):
    # Resize sesuai ukuran input model
    img = img.resize(TARGET_SIZE)

    # Konversi ke array
    img_array = np.array(img)

    # Jika gambar grayscale, ubah menjadi RGB
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)

    # Tambahkan batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Normalisasi
    img_array = img_array / 255.0

    # Prediksi
    prediction = model.predict(img_array)

    probability = float(prediction[0][0])

    if probability > 0.5:
        predicted_class = 1
    else:
        predicted_class = 0

    return CLASS_LABELS[predicted_class], probability

# ======================================================
# UPLOAD FILE
# ======================================================
uploaded_file = st.file_uploader(
    "Upload gambar",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Gambar yang diupload",
        use_container_width=True
    )

    if st.button("Prediksi"):

        with st.spinner("Melakukan prediksi..."):
            label, probability = predict_image(image, model)

        st.subheader("Hasil Prediksi")

        if label == "Positive":
            st.success(f"Hasil: {label}")
        else:
            st.warning(f"Hasil: {label}")

        st.write(f"Probabilitas Output Model: **{probability*100:.2f}%**")

        if probability > 0.5:
            st.write(
                f"Tingkat keyakinan kelas Positive: **{probability*100:.2f}%**"
            )
        else:
            st.write(
                f"Tingkat keyakinan kelas Negative: **{(1-probability)*100:.2f}%**"
            )
