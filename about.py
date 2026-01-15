import streamlit as st
import pandas as pd

def about_dataset():
    st.subheader("Tentang Dataset")

    col1, col2 = st.columns(2)

    # ===============================
    # Kolom Gambar
    # ===============================
    with col1:
        img_url = "https://upload.wikimedia.org/wikipedia/commons/8/82/Air_pollution.jpg"
        st.image(
            img_url,
            caption="Health & Environmental Dataset",
            use_container_width=True
        )

    # ===============================
    # Kolom Penjelasan
    # ===============================
    with col2:
        st.write(
            "Dataset yang digunakan dalam penelitian ini terdiri dari dua sumber utama, "
            "yaitu **Cardiovascular Disease Dataset** dan **UCI Air Quality Dataset**. "
            "Kedua dataset ini digunakan untuk menganalisis hubungan antara kondisi kesehatan "
            "kardiovaskular dan kualitas lingkungan udara.\n\n"

            "**Cardiovascular Disease Dataset** berisi data pasien yang mencakup usia, "
            "jenis kelamin, tekanan darah, kadar kolesterol, kebiasaan merokok, "
            "serta indikator medis lainnya yang digunakan untuk memprediksi risiko "
            "penyakit kardiovaskular.\n\n"

            "**UCI Air Quality Dataset** berisi data pengukuran kualitas udara seperti "
            "konsentrasi CO, NO2, NOx, dan C6H6, serta variabel lingkungan lainnya yang "
            "direkam secara berkala. Dataset ini merepresentasikan kondisi lingkungan "
            "yang berpotensi memengaruhi kesehatan manusia.\n\n"

            "Kombinasi kedua dataset ini memungkinkan analisis prediktif menggunakan "
            "metode *Machine Learning* seperti **Logistic Regression, Decision Tree, "
            "Random Forest, SVM, dan CatBoost** untuk memahami pola dan faktor risiko "
            "penyakit kardiovaskular berdasarkan data kesehatan dan lingkungan."
        )
