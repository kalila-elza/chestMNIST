import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Konfigurasi Halaman
st.set_page_config(page_title="Chest X-Ray Classifier", page_icon="🫁", layout="centered")

# --- Fungsi untuk memuat model ---
@st.cache_resource
def load_model():
    # Pastikan nama file sesuai dengan model yang Anda simpan
    model = tf.keras.models.load_model('model_chestnet.keras')
    return model

model = load_model()

# --- Judul dan Deskripsi ---
st.title("🫁 ChestMNIST X-Ray Classifier")
st.markdown("""
**Tugas Besar Sistem Biomedik Cerdas - Kelompok 7**
Aplikasi ini menggunakan model Convolutional Neural Network (ChestNet) untuk mendeteksi abnormalitas pada citra X-Ray dada.
""")

st.write("---")

# --- Upload Gambar ---
uploaded_file = st.file_uploader("Unggah gambar X-Ray Dada (Format JPG/PNG/JPEG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Tampilkan gambar yang diunggah
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar X-Ray yang diunggah', use_column_width=True)
    
    # --- Tombol Prediksi ---
    if st.button("Analisis X-Ray"):
        with st.spinner("Memproses gambar..."):
            # 1. Preprocessing Gambar (Sesuaikan dengan IMG_SIZE model Anda, misal 64x64)
            img = image.convert('L') # Convert to Grayscale
            img = img.resize((64, 64)) # Resize ke ukuran input model
            
            # Convert ke array Numpy dan normalisasi
            img_array = np.array(img) / 255.0
            
            # Tambahkan dimensi batch dan channel: (1, 64, 64, 1)
            img_array = np.expand_dims(img_array, axis=-1)
            img_array = np.expand_dims(img_array, axis=0)
            
            # 2. Prediksi
            predictions = model.predict(img_array)
            prob_normal = predictions[0][0]
            prob_abnormal = predictions[0][1]
            
            # 3. Hasil Prediksi
            st.write("### Hasil Diagnosis:")
            
            # Ambil index dengan probabilitas tertinggi
            predicted_class = np.argmax(predictions, axis=-1)[0]
            
            if predicted_class == 0:
                st.success(f"**NORMAL** (Kepercayaan: {prob_normal*100:.2f}%)")
            else:
                st.error(f"**ABNORMAL** (Kepercayaan: {prob_abnormal*100:.2f}%)")
            
            # Tampilkan progress bar probabilitas
            st.write("**Detail Probabilitas:**")
            st.progress(float(prob_normal), text=f"Normal ({prob_normal*100:.1f}%)")
            st.progress(float(prob_abnormal), text=f"Abnormal ({prob_abnormal*100:.1f}%)")
