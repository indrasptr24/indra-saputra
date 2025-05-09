import streamlit as st
from utils import load_hama_model, predict_hama
import os

# Load model, label dan solusi
model = load_hama_model()
class_names = ['Belalang','Penggerek_Batang','Tikus_Sawah','Ulat_Grayak','Walang_Sangit','Wereng_Coklat']
solutions = {
    'Belalang':  'Pengendalian biologis dengan burung pemangsa atau semprot insektisida kontak.',
 'Penggerek_Batang': 'Gunakan insektisida sistemik yang dapat diserap oleh tanaman serta lakukan pemangkasan pada bagian tanaman yang terinfeksi.',
 'Tikus_Sawah': 'Buat perangkap tikus atau gunakan umpan racun tikus yang aman dan Gunakan predator alami seperti kucing atau Burung Hantu.',
 'Ulat_Grayak': 'Gunakan insektisida yang efektif terhadap ulat, seperti insektisida berbahan aktif Bacillus thuringiensis (Bt).',
 'Walang_Sangit': 'Penggunaan insektisida kontak saat fase bunting hingga pengisian bulir.' ,
 'Wereng_Coklat' : 'Gunakan insektisida yang ditujukan untuk wereng, seperti insektisida berbahan aktif imidacloprid.'
}

st.title("Klasifikasi Hama Tanaman Padi 🌾")
st.write("Upload gambar hama dan dapatkan hasil klasifikasinya.")

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Simpan gambar sementara
    img_path = os.path.join("images", uploaded_file.name)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(img_path, caption="Gambar Diupload", use_column_width=True)

    if st.button("Prediksi Hama"):
        label, confidence = predict_hama(img_path, model, class_names)
        st.success(f"Hama terdeteksi: **{label}** dengan keyakinan {confidence:.2f}")
        st.markdown("### Solusi Penanganan:")
        st.info(solusi_hama[label])

