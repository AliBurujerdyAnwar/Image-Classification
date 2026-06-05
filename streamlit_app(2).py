import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import requests

st.set_page_config(
    page_title="Image Classification",
    page_icon="🖼️",
    layout="wide"
)

# --- KONFIGURASI GOOGLE DRIVE ---
# Ganti ID di bawah ini dengan ID file .h5 Anda dari Google Drive
# Contoh link share: https://drive.google.com/file/d/1XyZ...abc/view?usp=sharing
# Ambil bagian string panjang di antara '/d/' dan '/view'
DRIVE_FILE_ID = "1p2SdclWeEJUPt_zn5m0Pz6DwwrxrqJHC"
MODEL_FILE = "load_image_classification_model"

@st.cache_resource
def load_model_from_drive(file_id, output_path):
    """Mengunduh model dari Google Drive jika belum ada, lalu memuatnya."""
    if not os.path.exists(output_path):
        with st.spinner("Mengunduh model dari Google Drive... Harap tunggu."):
            # URL untuk download langsung dari Google Drive API terbuka
            download_url = f"https://docs.google.com/uc?export=download&id={file_id}"
            
            # Melakukan request untuk mengunduh file
            response = requests.get(download_url, stream=True)
            if response.status_code == 200:
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            else:
                raise Exception(f"Gagal mengunduh file. Status code: {response.status_code}. Pastikan ID benar dan file di-share ke 'Anyone with the link'.")
                
    return tf.keras.models.load_model(output_path)

# Mencoba memuat model menggunakan fungsi baru
try:
    model = load_model_from_drive(DRIVE_FILE_ID, MODEL_FILE)
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# --- SISA KODE KLASIFIKASI (TETAP SAMA) ---
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
    "yang diunduh secara otomatis dari Google Drive."
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
st.caption("Dikonversi dari notebook Jupyter ke Streamlit dengan integrasi Google Drive")
