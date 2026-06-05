import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import zipfile
import io
import gc
import urllib.request

# 1. Konfigurasi Halaman Web
st.set_page_config(
    page_title="Image Classification via ZIP",
    page_icon="🖼️",
    layout="wide"
)

# Sesuaikan dengan nama file model asli Anda
MODEL_FILE = "load_image_classification_model.h5"

# =========================================================================
# PASTE LINK GOOGLE DRIVE KAMU DI BAWAH INI (Ganti teks di dalam tanda kutip)
# =========================================================================
GDrive_Link = "https://drive.google.com/file/d/1p2SdclWeEJUPt_zn5m0Pz6DwwrxrqJHC/view?usp=sharing"

# Fungsi untuk mengubah link share Google Drive menjadi link download langsung
def get_direct_download_link(url):
    if "drive.google.com" in url:
        if "/file/d/" in url:
            file_id = url.split("/file/d/")[1].split("/")[0]
            return f"https://docs.google.com/uc?export=download&id={file_id}"
    return url

@st.cache_resource
def load_model_from_drive():
    # Jika file model belum ada di server Streamlit, download dari Google Drive
    if not os.path.exists(MODEL_FILE):
        if GDrive_Link == "MASUKKAN_LINK_GOOGLE_DRIVE_KAMU_DI_SINI" or GDrive_Link == "":
            st.warning("⚠️ Kamu belum memasukkan link Google Drive di dalam kode app.py!")
            return None
        
        with st.spinner("⏳ Mengunduh file model h5 dari Google Drive (Hanya dilakukan sekali saat pertama kali dibuka)..."):
            try:
                direct_link = get_direct_download_link(GDrive_Link)
                urllib.request.urlretrieve(direct_link, MODEL_FILE)
                st.success("✅ Model berhasil diunduh dari Google Drive!")
            except Exception as e:
                st.error(f"❌ Gagal mengunduh model dari Google Drive: {e}")
                return None
                
    try:
        return tf.keras.models.load_model(MODEL_FILE)
    except Exception as e:
        st.error(f"Gagal memuat file model h5: {e}")
        return None

# Memuat model
model = load_model_from_drive()

# TARGET SIZE ASLI DARI NOTEBOOK ANDA (128x128)
target_size = (128, 128)
class_labels = {0: "negative", 1: "positive"}

st.title("🔍 Concrete Crack Image Classification via ZIP")
st.write("Unggah berkas **.zip** berisi kumpulan foto beton untuk mendeteksi kategori secara otomatis.")

if model is not None:
    st.success("✅ Model AI berhasil dimuat dan siap digunakan!")

# Input dikunci khusus untuk berkas ZIP saja
uploaded_zip = st.file_uploader("Pilih dan unggah file ZIP berisi kumpulan foto...", type=["zip"])

if uploaded_zip is not None and model is not None:
    st.write("---")
    st.info("📦 Berkas ZIP terdeteksi! Mengekstrak isi file...")
    
    try:
        with zipfile.ZipFile(uploaded_zip) as z:
            all_files = z.namelist()
            valid_extensions = ('.jpg', '.jpeg', '.png')
            
            # Memfilter hanya file foto asli dan membuang file sampah sistem
            image_files = [
                f for f in all_files 
                if f.lower().endswith(valid_extensions) 
                and not f.startswith('__MACOSX/') 
                and not os.path.basename(f).startswith('.')
            ]
            
            total_images = len(image_files)
            
            if total_images == 0:
                st.warning("⚠️ Tidak ditemukan file gambar (.jpg/.png) yang valid di dalam ZIP Anda.")
            else:
                st.success(f"🚀 Menemukan {total_images} gambar. Memulai klasifikasi otomatis...")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                results = []

                # Proses gambar satu per satu dari file ZIP
                for idx, file_name in enumerate(image_files):
                    status_text.text(f"Menganalisis ({idx + 1}/{total_images}): {os.path.basename(file_name)}")
                    
                    try:
                        img_data = z.read(file_name)
                        with Image.open(io.BytesIO(img_data)) as img:
                            # Preprocessing persis seperti file notebook asli Anda
                            img = img.convert("RGB")
                            img = img.resize(target_size)
                            img_array = np.array(img).astype("float32") / 255.0
                            img_array = np.expand_dims(img_array, axis=0)
                        
                        # Menghitung Prediksi
                        prediction = model.predict(img_array, verbose=0)
                        probability = float(prediction[0][0])

                        if probability > 0.5:
                            predicted_class = "Positive (Retak)"
                            confidence = probability
                        else:
                            predicted_class = "Negative (Aman)"
                            confidence = 1 - probability
                        
                        results.append({
                            "Nama File": os.path.basename(file_name),
                            "Prediksi": predicted_class,
                            "Tingkat Keyakinan": f"{confidence * 100:.2f}%",
                            "Raw Probability": f"{probability:.4f}"
                        })
                        
                    except Exception:
                        continue
                    
                    # Update Progress Bar
                    progress_bar.progress((idx + 1) / total_images)
                    
                    # Pembersihan RAM berkala agar server tidak crash
                    if (idx + 1) % 5 == 0:
                        tf.keras.backend.clear_session()
                        gc.collect()

                status_text.empty()
                
                # Menampilkan output tabel hasil akhir jika berhasil diproses
                if results:
                    st.write("### 📊 Hasil Klasifikasi Keseluruhan:")
                    st.dataframe(results, use_container_width=True)
                    
                    total_pos = sum(1 for r in results if "Positive" in r["Prediksi"])
                    total_neg = sum(1 for r in results if "Negative" in r["Prediksi"])
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Total Kategori Positive (Retak)", total_pos)
                    col2.metric("Total Kategori Negative (Aman)", total_neg)
                        
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat membaca file ZIP: {e}")
