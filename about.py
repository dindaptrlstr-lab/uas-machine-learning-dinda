import streamlit as st
import pandas as pd

def show_about():

    # =========================
    # PENGAMAN (SESSION STATE)
    # =========================
    if "df" not in st.session_state:
        st.warning("Silakan unggah dan pilih dataset terlebih dahulu pada menu Unggah Data.")
        return

    df = st.session_state["df"]
    dataset_name = st.session_state.get("dataset_name", "Tidak diketahui")

    # =========================
    # DESKRIPSI UMUM DATASET
    # =========================
    st.subheader("Informasi Umum Dataset")

    if dataset_name == "water_potability.csv":
        st.markdown("""
        **Dataset Kelayakan Air Minum (Water Potability Dataset)** digunakan untuk
        melakukan **klasifikasi kelayakan air minum**
        berdasarkan parameter kualitas air secara fisik dan kimia.

        **Fitur yang digunakan:**
        - pH  
        - Hardness  
        - Solids  
        - Chloramines  
        - Sulfate  
        - Conductivity  
        - Organic Carbon  
        - Trihalomethanes  
        - Turbidity  

        **Variabel Target:**
        - `Potability`  
          - 0 → Air tidak layak konsumsi  
          - 1 → Air layak konsumsi  

        **Jenis Permasalahan:**
        - Supervised Learning  
        - Klasifikasi Biner
        """)
        dataset_type = "Lingkungan"

    elif dataset_name == "cardio_train.csv":
        st.markdown("""
        **Dataset Penyakit Kardiovaskular (Cardiovascular Disease Dataset)** digunakan untuk
        memprediksi **risiko penyakit kardiovaskular**
        berdasarkan data klinis dan gaya hidup pasien.

        **Fitur yang digunakan:**
        - Usia  
        - Jenis Kelamin  
        - Tekanan Darah  
        - Kolesterol  
        - Glukosa  
        - Indeks Massa Tubuh (BMI)  
        - Kebiasaan Merokok  
        - Aktivitas Fisik  

        **Variabel Target:**
        - `cardio`  
          - 0 → Tidak berisiko  
          - 1 → Berisiko  

        **Jenis Permasalahan:**
        - Supervised Learning  
        - Klasifikasi Biner
        """)
        dataset_type = "Kesehatan"

    else:
        st.markdown("""
        Dataset diunggah oleh pengguna.

        Informasi mengenai variabel target dan jenis permasalahan
        akan ditentukan pada tahap **Pemodelan Machine Learning**.
        """)
        dataset_type = "Tidak diketahui"

    # =========================
    # RINGKASAN DATASET
    # =========================
    st.subheader("Ringkasan Dataset")

    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Data", df.shape[0])
    col2.metric("Jumlah Fitur", df.shape[1])
    col3.metric("Kategori Dataset", dataset_type)

    # =========================
    # PRATINJAU DATA
    # =========================
    st.subheader("Pratinjau Data")
    st.dataframe(df.head(), use_container_width=True)

    # =========================
    # METODE MACHINE LEARNING
    # =========================
    st.subheader("Metode Machine Learning yang Digunakan")

    st.markdown("""
    Aplikasi ini menerapkan beberapa algoritma
    **Machine Learning untuk klasifikasi**, yaitu:

    - Logistic Regression  
    - Decision Tree Classifier  
    - Random Forest Classifier  
    - Support Vector Machine (SVM)  
    - CatBoost Classifier  

    Model-model tersebut digunakan untuk
    **membandingkan kinerja prediksi**
    menggunakan metrik evaluasi berikut:
    - Akurasi (Accuracy)  
    - Presisi (Precision)  
    - Recall  
    - F1-Score  
    - ROC-AUC
    """)

    # =========================
    # CATATAN PRA-PROSES DATA
    # =========================
    st.info(
        "Catatan:\n"
        "- Dataset akan melalui tahap pra-pemrosesan sebelum dilakukan pemodelan.\n"
        "- Pra-pemrosesan meliputi penanganan nilai hilang dan standarisasi fitur.\n"
        "- Variabel target dapat ditentukan secara otomatis atau pada menu Machine Learning.\n"
        "- Hasil evaluasi model ditampilkan pada menu Machine Learning."
    )
