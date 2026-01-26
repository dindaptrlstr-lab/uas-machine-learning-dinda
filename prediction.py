import streamlit as st
import pandas as pd
import numpy as np


def prediction_page():
    st.subheader("Prediction App")

    # =========================
    # DESKRIPSI HALAMAN
    # =========================
    st.markdown("""
    Halaman ini digunakan untuk melakukan **prediksi data baru**
    menggunakan **model terbaik** yang diperoleh dari proses training
    pada menu **Machine Learning**. Model dipilih berdasarkan **F1-Score terbaik** dan digunakan kembali
    secara konsisten untuk proses inferensi.
    """)
    st.markdown("---")

    # =========================
    # PENGAMAN PIPELINE
    # =========================
    if "best_model" not in st.session_state:
        st.warning("Silakan jalankan menu **Machine Learning** terlebih dahulu.")
        return

    if "df" not in st.session_state or "dataset_name" not in st.session_state:
        st.warning("Silakan upload dataset terlebih dahulu melalui sidebar.")
        return

    if "feature_columns" not in st.session_state:
        st.warning("Informasi fitur tidak tersedia. Silakan jalankan training ulang.")
        return

    # =========================
    # AMBIL OBJEK DARI SESSION
    # =========================
    model = st.session_state["best_model"]
    df = st.session_state["df"]
    dataset_name = st.session_state["dataset_name"]
    feature_columns = st.session_state["feature_columns"]
    scaler = st.session_state.get("scaler")

    # =========================
    # TARGET & LABEL OTOMATIS
    # =========================
    if dataset_name == "water_potability.csv":
        target_col = "Potability"
        positive_label = "Layak Minum"
        negative_label = "Tidak Layak Minum"

    elif dataset_name == "cardio_train.csv":
        target_col = "cardio"
        positive_label = "Berisiko Penyakit Jantung"
        negative_label = "Tidak Berisiko"

    else:
        st.error("Dataset tidak dikenali.")
        return

    st.write(
        "Prediksi dilakukan menggunakan **model terbaik** "
        "yang telah dilatih sebelumnya."
    )

    st.markdown("---")

    # =========================
    # INPUT DATA OTOMATIS
    # =========================
    st.subheader("Data Input")

    input_df = df[feature_columns].iloc[-1:].copy()

    st.write(
        "Data input diambil dari **baris terakhir dataset** "
        "sebagai contoh observasi baru untuk prediksi."
    )

    st.dataframe(input_df, use_container_width=True)

    # =========================
    # PREPROCESSING KONSISTEN
    # =========================
    if scaler is not None:
        input_processed = scaler.transform(input_df)
    else:
        input_processed = input_df.values

    # =========================
    # JALANKAN PREDIKSI
    # =========================
    if st.button("🔍 Jalankan Prediksi"):

        prediction = model.predict(input_processed)[0]

        st.markdown("---")
        st.subheader("Hasil Prediksi")

        if prediction == 1:
            st.success(f"✅ **{positive_label}**")
        else:
            st.error(f"❌ **{negative_label}**")

        st.write(
            "Hasil ini diperoleh dari **model terbaik** "
            "berdasarkan evaluasi **F1-Score** "
            "pada tahap Machine Learning."
        )

    # =========================
    # CATATAN AKADEMIK
    # =========================
    st.markdown("---")
    st.info(
        "Catatan:\n"
        "- Prediksi bersifat **klasifikasi**, bukan prediksi waktu.\n"
        "- Data input berasal dari dataset yang di-upload.\n"
        "- Hasil prediksi digunakan untuk **analisis dan pembelajaran**.\n"
        "- Model tidak dimaksudkan sebagai alat diagnosis medis."
    )

