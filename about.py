import streamlit as st
import pandas as pd

def show_about():
    st.title("📘 About Dataset")

    # =========================
    # PENGAMAN
    # =========================
    if "df" not in st.session_state:
        st.warning("Silakan upload dataset terlebih dahulu.")
        return

    df = st.session_state["df"]
    dataset_name = st.session_state.get("dataset_name", "Tidak diketahui")

    # =========================
    # DESKRIPSI DATASET
    # =========================
    st.subheader("📂 Informasi Umum Dataset")

    if dataset_name == "water_potability.csv":
        st.write("""
        **Water Potability Dataset** digunakan untuk
        mengklasifikasikan apakah air **layak dikonsumsi (potable)**
        atau **tidak layak** berdasarkan parameter kualitas air.

        **Contoh fitur utama:**
        - pH
        - Hardness
        - Solids
        - Chloramines
        - Sulfate
        - Conductivity
        - Organic Carbon
        - Trihalomethanes
        - Turbidity

        **Target:**
        - `Potability` (0 = Tidak Layak, 1 = Layak)
        """)

        dataset_type = "Lingkungan"

    elif dataset_name == "cardio_train.csv":
        st.write("""
        **Cardiovascular Disease Dataset** digunakan untuk
        memprediksi **risiko penyakit kardiovaskular**
        berdasarkan data klinis pasien.

        **Contoh fitur utama:**
        - Usia
        - Jenis Kelamin
        - Tekanan Darah
        - Kolesterol
        - Glukosa
        - BMI
        - Kebiasaan merokok
        - Aktivitas fisik

        **Target:**
        - `cardio` (0 = Tidak Berisiko, 1 = Berisiko)
        """)

        dataset_type = "Kesehatan"

    else:
        st.write("Dataset diunggah oleh pengguna.")
        dataset_type = "Tidak diketahui"

    # =========================
    # RINGKASAN DATASET
    # =========================
    st.subheader("📊 Ringkasan Dataset")

    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Data", df.shape[0])
    col2.metric("Jumlah Fitur", df.shape[1])
    col3.metric("Jenis Dataset", dataset_type)

    # =========================
    # INFORMASI KOLOM
    # =========================
    st.subheader("🧱 Struktur Dataset")

    column_info = pd.DataFrame({
        "Nama Kolom": df.columns,
        "Tipe Data": df.dtypes.astype(str)
    })

    st.dataframe(column_info, use_container_width=True)

    # =========================
    # CONTOH DATA
    # =========================
    st.subheader("🔍 Contoh Data")
    st.dataframe(df.head(), use_container_width=True)

    # =========================
    # METODE MACHINE LEARNING
    # =========================
    st.subheader("🤖 Metode Machine Learning yang Digunakan")

    st.markdown("""
    Model klasifikasi yang digunakan dalam aplikasi ini:

    - **Logistic Regression**
    - **Decision Tree**
    - **Random Forest**
    - **Support Vector Machine (SVM)**
    - **CatBoost Classifier**

    Model-model ini digunakan untuk membandingkan performa prediksi
    berdasarkan akurasi, precision, recall, F1-score, dan ROC-AUC.
    """)

    # =========================
    # CATATAN
    # =========================
    st.info(
        "Catatan:\n"
        "- Dataset dipilih otomatis berdasarkan file yang di-upload.\n"
        "- Target klasifikasi akan digunakan pada menu **Machine Learning**.\n"
        "- Data akan melalui preprocessing sebelum pemodelan."
    )
