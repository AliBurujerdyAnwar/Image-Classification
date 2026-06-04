import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Concrete Crack Classification",
    page_icon="🔍",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.big-font {
    font-size:24px !important;
    font-weight:bold;
    color:#1E88E5;
}

.result-positive {
    padding:15px;
    border-radius:10px;
    background-color:#d4edda;
    color:#155724;
    font-size:20px;
    font-weight:bold;
}

.result-negative {
    padding:15px;
    border-radius:10px;
    background-color:#f8d7da;
    color:#721c24;
    font-size:20px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL
# =====================================================

MODEL_PATH = "load_image_classification_model.h5"

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.title("⚙️ Informasi Sistem")

    st.success("Model berhasil dimuat")

    st.markdown("---")

    st.write("### Parameter Model")

    st.write("Ukuran Input : 128 x 128")
    st.write("Jumlah Kelas : 2")
    st.write("Kelas:")
    st.write("- Positive (Crack)")
    st.write("- Negative (No Crack)")

    st.markdown("---")

    st.info(
        """
        Aplikasi ini digunakan untuk
        mengklasifikasikan citra beton
        menjadi:

        ✔ Positive (Retak)

        ✔ Negative (Tidak Retak)
        """
    )

# =====================================================
# HEADER
# =====================================================

st.markdown(
    "<p class='big-font'>🔍 Concrete Crack Classification System</p>",
    unsafe_allow_html=True
)

st.write(
    """
    Upload gambar beton untuk mendeteksi apakah terdapat
    retakan (Positive) atau tidak terdapat retakan (Negative).
    """
)

# =====================================================
# FUNCTION
# =====================================================

IMG_SIZE = (128, 128)

def predict_image(img):

    img = img.resize(IMG_SIZE)

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array / 255.0

    prediction = model.predict(img_array, verbose=0)

    positive_prob = float(prediction[0][0])

    negative_prob = 1 - positive_prob

    if positive_prob >= 0.5:
        label = "Positive"
        confidence = positive_prob
    else:
        label = "Negative"
        confidence = negative_prob

    return label, confidence, positive_prob, negative_prob

# =====================================================
# SESSION HISTORY
# =====================================================

if "history" not in st.session_state:
    st.session_state.history = []

# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload Gambar",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image_pil = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:

        st.image(
            image_pil,
            caption="Gambar Input",
            use_container_width=True
        )

    with col2:

        with st.spinner("Menganalisis gambar..."):

            label, confidence, positive_prob, negative_prob = predict_image(image_pil)

        st.subheader("📊 Hasil Prediksi")

        if label == "Positive":

            st.markdown(
                f"""
                <div class='result-positive'>
                Positive (Crack Detected)
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class='result-negative'>
                Negative (No Crack)
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("### Tingkat Keyakinan")

        st.progress(float(confidence))

        st.metric(
            "Confidence",
            f"{confidence*100:.2f}%"
        )

        # save history
        st.session_state.history.append({
            "Prediction": label,
            "Confidence (%)": round(confidence*100,2)
        })

# =====================================================
# PROBABILITY CHART
# =====================================================

        st.write("### Distribusi Probabilitas")

        df = pd.DataFrame({
            "Class": ["Negative", "Positive"],
            "Probability": [
                negative_prob*100,
                positive_prob*100
            ]
        })

        fig, ax = plt.subplots(figsize=(5,3))

        ax.bar(
            df["Class"],
            df["Probability"]
        )

        ax.set_ylabel("Probability (%)")
        ax.set_ylim([0,100])

        st.pyplot(fig)

# =====================================================
# DETAIL HASIL
# =====================================================

        st.write("### Detail Probabilitas")

        col_a, col_b = st.columns(2)

        with col_a:
            st.metric(
                "Negative",
                f"{negative_prob*100:.2f}%"
            )

        with col_b:
            st.metric(
                "Positive",
                f"{positive_prob*100:.2f}%"
            )

# =====================================================
# HISTORY
# =====================================================

if len(st.session_state.history) > 0:

    st.markdown("---")

    st.subheader("📝 Riwayat Prediksi")

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "Image Classification using Deep Learning | "
    "Concrete Crack Detection Project"
)
