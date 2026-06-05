import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# =====================================================
# KONFIGURASI
# =====================================================

st.set_page_config(
    page_title="Concrete Crack Classification",
    page_icon="🔍",
    layout="wide"
)

# =====================================================
# GOOGLE DRIVE MODEL
# =====================================================

# Ganti dengan ID file Google Drive Anda
FILE_ID = "https://drive.google.com/file/d/1p2SdclWeEJUPt_zn5m0Pz6DwwrxrqJHC/view?usp=sharing"

MODEL_PATH = "model.h5"

@st.cache_resource
def load_model():

    # Download model jika belum ada
    if not os.path.exists(MODEL_PATH):

        url = f"https://drive.google.com/uc?id={https://drive.google.com/file/d/1p2SdclWeEJUPt_zn5m0Pz6DwwrxrqJHC/view?usp=sharing}"

        with st.spinner("Mengunduh model dari Google Drive..."):
            gdown.download(
                url,
                MODEL_PATH,
                quiet=False
            )

    model = tf.keras.models.load_model(MODEL_PATH)

    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# =====================================================
# PARAMETER MODEL
# =====================================================

IMG_SIZE = (128, 128)

CLASS_NAMES = [
    "Negative",
    "Positive"
]

# =====================================================
# FUNGSI PREDIKSI
# =====================================================

def predict_image(image):

    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    img_array = np.array(image)

    img_array = img_array.astype(np.float32) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)

    probability = float(prediction[0][0])

    if probability >= 0.5:
        label = CLASS_NAMES[1]
        confidence = probability
    else:
        label = CLASS_NAMES[0]
        confidence = 1 - probability

    return label, confidence, probability

# =====================================================
# STREAMLIT UI
# =====================================================

st.title("🔍 Concrete Crack Classification")

uploaded_file = st.file_uploader(
    "Upload gambar",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Input Image",
            use_container_width=True
        )

    with col2:

        label, confidence, probability = predict_image(image)

        st.subheader("Hasil Prediksi")

        if label == "Positive":
            st.success(f"✅ {label}")
        else:
            st.info(f"❌ {label}")

        st.metric(
            "Confidence",
            f"{confidence*100:.2f}%"
        )

        st.progress(confidence)

        st.write(
            f"Raw Probability: {probability:.4f}"
        )
