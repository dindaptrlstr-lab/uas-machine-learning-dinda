import streamlit as st
import pandas as pd
import numpy as np


def prediction_page():
    st.subheader("Sistem Prediksi Risiko Penyakit Kardiovaskular")

    # =========================
    # DESKRIPSI HALAMAN
    # =========================
    st.markdown("""
    Halaman ini digunakan untuk memasukkan data kesehatan pengguna 
    dan memperoleh hasil prediksi risiko penyakit kardiovaskular.
    """)
    st.markdown("---")

    # =========================
    # PENGAMAN PIPELINE
    # =========================
    if "best_model" not in st.session_state:
        st.warning("Silakan jalankan menu **Machine Learning** terlebih dahulu.")
        return

    if "feature_columns" not in st.session_state:
        st.warning("Informasi fitur tidak tersedia. Silakan lakukan training ulang.")
        return

    # =========================
    # AMBIL OBJEK SESSION
    # =========================
    model = st.session_state["best_model"]
    scaler = st.session_state.get("scaler", None)
    feature_columns = st.session_state["feature_columns"]
    dataset_name = st.session_state.get("dataset_name", "")

    # =========================
    # LABEL TARGET
    # =========================
    if dataset_name == "cardio_train.csv":
        positive_label = "Berisiko Penyakit Jantung"
        negative_label = "Tidak Berisiko"
    else:
        positive_label = "Risiko Tinggi"
        negative_label = "Risiko Rendah"

    # =========================
    # INPUT DATA
    # =========================
    st.subheader("Input Data")
    col1, col2 = st.columns(2)

    # Umur (tahun → hari)
    age_years = col1.number_input(
        "Umur (tahun)", min_value=10, max_value=100, value=50, step=1
    )
    age = age_years * 365

    # Jenis Kelamin
    gender_label = col2.selectbox("Jenis Kelamin", ["Pria", "Wanita"])
    gender = 1 if gender_label == "Pria" else 2

    # Tinggi & Berat
    height = col1.number_input(
        "Tinggi Badan (cm)", min_value=100, max_value=220, value=165, step=1
    )
    weight = col2.number_input(
        "Berat Badan (kg)", min_value=30, max_value=200, value=70, step=1
    )

    # Tekanan Darah
    ap_hi = col1.number_input(
        "Tekanan Darah Sistolik", min_value=80, max_value=250, value=120
    )
    ap_lo = col2.number_input(
        "Tekanan Darah Diastolik", min_value=50, max_value=150, value=80
    )

    # Kolesterol
    cholesterol_label = col1.selectbox(
        "Kolesterol", ["Normal", "Tinggi", "Sangat Tinggi"]
    )
    cholesterol = {"Normal": 1, "Tinggi": 2, "Sangat Tinggi": 3}[cholesterol_label]

    # Gula Darah
    gluc_label = col2.selectbox(
        "Gula Darah", ["Normal", "Tinggi", "Sangat Tinggi"]
    )
    gluc = {"Normal": 1, "Tinggi": 2, "Sangat Tinggi": 3}[gluc_label]

    # Kebiasaan
    smoke = 1 if col1.selectbox("Merokok", ["Tidak", "Ya"]) == "Ya" else 0
    alco = 1 if col2.selectbox("Konsumsi Alkohol", ["Tidak", "Ya"]) == "Ya" else 0
    active = 1 if col2.selectbox("Aktivitas Fisik", ["Tidak Aktif", "Aktif"]) == "Aktif" else 0

    # =========================
    # DATAFRAME INPUT DASAR
    # =========================
    input_df = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "height": height,
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active
    }])

    # =========================
    # PENYESUAIAN FITUR (ANTI KEYERROR)
    # =========================
    # Tambahkan kolom yang ada di training tapi tidak ada di input
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0  # default aman

    # Ambil hanya fitur yang dipakai model & urutkan
    input_df = input_df[feature_columns]

    # =========================
    # PREPROCESSING
    # =========================
    if scaler is not None:
        input_processed = scaler.transform(input_df)
    else:
        input_processed = input_df.values

    # =========================
    # PREDIKSI
    # =========================
    if st.button("🔍 Jalankan Prediksi"):
        prediction = model.predict(input_processed)[0]

        st.markdown("---")
        st.subheader("Hasil Prediksi")

        if prediction == 1:
            st.error(f"⚠️ **{positive_label}**")
        else:
            st.success(f"✅ **{negative_label}**")

    # =========================
    # CATATAN
    # =========================
    st.markdown("---")
    st.info(
        "Catatan:\n"
        "- Sistem ini digunakan untuk **pembelajaran dan analisis data**, "
        "bukan sebagai diagnosis medis."
    )

