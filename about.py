import streamlit as st
import pandas as pd

def about_dataset():
    st.write('## Tentang Dataset')
    col1, col2 = st.columns([5,5])

    with col1:
        link = "https://myhealthcentre.ca/wp-content/uploads/2025/03/17096-819x583.jpg" 
        st.image(link, caption="Health & Environmental Dataset")

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
            "konsentrasi CO, NO2, O3, PM10, serta variabel lingkungan lainnya yang "
            "direkam secara berkala. Dataset ini digunakan untuk merepresentasikan "
            "kondisi lingkungan yang berpotensi memengaruhi kesehatan manusia.\n\n"

            "Kombinasi kedua dataset ini memungkinkan analisis prediktif menggunakan "
            "metode *Machine Learning* seperti **Logistic Regression, Decision Tree, "
            "Random Forest, SVM, dan CatBoost** untuk memahami pola dan faktor risiko "
            "penyakit kardiovaskular berdasarkan data kesehatan dan lingkungan."
        )
