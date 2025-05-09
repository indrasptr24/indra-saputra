import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import gdown

def load_hama_model():
    model_path = "model/model_hama.h5"
    os.makedirs("model", exist_ok=True)
    url = "https://drive.google.com/uc?id=1ahp6x3MDsBLz_M0vcRi6uq64XYlNb_TL"
    gdown.download(url, model_path, quiet=False)

    if not os.path.exists(model_path):
        st.error("File model tidak ditemukan setelah diunduh!")
    else:
        st.success("File model berhasil diunduh.")
    return load_model(model_path)

def predict_hama(img_path, model, class_names):
    img = image.load_img(img_path, target_size=(200, 200))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_class, confidence

st.title("Klasifikasi Hama Tanaman Padi")

# File uploader untuk gambar
uploaded_file = st.file_uploader("Upload gambar hama", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Simpan file gambar sementara
    os.makedirs("images", exist_ok=True)
    
    img_path = os.path.join("images", uploaded_file.name)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Menampilkan gambar yang diupload
    st.image(img_path, caption="Gambar yang diupload", use_column_width=True)

    # Menyiapkan model dan memprediksi
    model = load_hama_model()
    class_names = ['Belalang', 'Penggerek_Batang', 'Tikus_Sawah', 'Ulat_Grayak', 'Walang_Sangit', 'Wereng_Coklat']
    solutions = {
        'Belalang': 'Pengendalian biologis dengan burung pemangsa atau semprot insektisida kontak.',
        'Penggerek_Batang': 'Gunakan insektisida sistemik yang dapat diserap oleh tanaman serta lakukan pemangkasan pada bagian tanaman yang terinfeksi.',
        'Tikus_Sawah': 'Buat perangkap tikus atau gunakan umpan racun tikus yang aman dan Gunakan predator alami seperti kucing atau Burung Hantu.',
        'Ulat_Grayak': 'Gunakan insektisida yang efektif terhadap ulat, seperti insektisida berbahan aktif Bacillus thuringiensis (Bt).',
        'Walang_Sangit': 'Penggunaan insektisida kontak saat fase bunting hingga pengisian bulir.',
        'Wereng_Coklat': 'Gunakan insektisida yang ditujukan untuk wereng, seperti insektisida berbahan aktif imidacloprid.'
    }

    predicted_class, confidence = predict_hama(img_path, model, class_names)

    # Menampilkan hasil prediksi
    st.write(f"**Prediksi:** {predicted_class}")
    st.write(f"**Tingkat Keyakinan:** {confidence * 100:.2f}%")
    if predicted_class in solutions:
        st.info(f"**Solusi Pengendalian:** {solutions[predicted_class]}")
    else:
        st.warning("Solusi belum tersedia untuk kelas ini.")
