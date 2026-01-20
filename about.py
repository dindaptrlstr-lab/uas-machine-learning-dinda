import streamlit as st
import pandas as pd

def about_dataset():
    st.subheader("Tentang Dataset")

    col1, col2 = st.columns(2)

    # ===============================
    # Kolom Gambar
    # ===============================
    with col1:
        img_url = "https://upload.wikimedia.org/wikipedia/commons/3/3e/Drinking_water.jpg"
        st.image(
            img_url,
            caption="Health & Water Quality Dataset",
            use_container_width=True
        )

    # ===============================
    # Kolom Penjelasan
    # ===============================
    with col2:
        st.write(
            "Dataset yang digunakan dalam penelitian ini terdiri dari dua sumber utama, "
            "yaitu **Cardiovascular Disease Dataset** dan **Water Potability Dataset**. "
            "Kedua dataset ini digunakan untuk menganalisis hubungan antara kondisi kesehatan "
            "kardiovaskular dan kualitas lingkungan air.\n\n"

            "**Cardiovascular Disease Dataset** berisi data pasien yang mencakup usia, "
            "jenis kelamin, tekanan darah, kadar kolesterol, kebiasaan merokok, "
            "serta indikator medis lainnya yang digunakan untuk memprediksi risiko "
            "penyakit kardiovaskular.\n\n"

            "**Water Potability Dataset** berisi data kualitas air yang mencakup "
            "parameter fisik dan kimia seperti **pH, Hardness, Total Dissolved Solids, "
            "Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, "
            "dan Turbidity**, dengan variabel target **Potability** yang menunjukkan "
            "kelayakan air untuk dikonsumsi.\n\n"

            "Kombinasi kedua dataset ini memungkinkan analisis prediktif menggunakan "
            "metode *Machine Learning* seperti **Logistic Regression, Decision Tree, "
            "Random Forest, SVM, dan CatBoost** untuk memahami pengaruh faktor kesehatan "
            "dan lingkungan terhadap risiko penyakit kardiovaskular serta kualitas "
            "hidup masyarakat."
        )
