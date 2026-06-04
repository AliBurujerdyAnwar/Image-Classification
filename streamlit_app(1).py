import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Image Classification",
    page_icon="🖼️",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("load_image_classification_model.h5")

st.title("🖼️ Image Classification")
st.markdown("Upload gambar untuk klasifikasi **negative** atau **positive**.")

try:
    model = load_model()
    st.success("Model berhasil dimuat.")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

TARGET_SIZE = (128, 128)
CLASS_LABELS = {0: "negative", 1: "positive"}

uploaded_file = st.file_uploader(
    "Pilih gambar",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Gambar Input", use_container_width=True)

    img_resized = image.resize(TARGET_SIZE)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)
    probability = float(prediction[0][0])

    predicted_index = 1 if probability > 0.5 else 0
    predicted_label = CLASS_LABELS[predicted_index]

    confidence = probability if predicted_index == 1 else (1 - probability)

    with col2:
        st.subheader("Hasil Prediksi")
        st.metric("Kelas", predicted_label)
        st.metric("Confidence", f"{confidence*100:.2f}%")

    st.progress(float(confidence))

    st.write("### Detail Probabilitas")
    st.write({
        "negative": round((1 - probability) * 100, 2),
        "positive": round(probability * 100, 2)
    })

st.markdown("---")
st.caption("Pastikan file model bernama: load_image_classification_model.h5 dan berada pada folder yang sama dengan streamlit_app.py")
