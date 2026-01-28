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
    pada menu **Machine Learning**.  
    Data yang digunakan untuk prediksi **diinput secara manual**
    dan **bukan berasal dari dataset pelatihan**.
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
        positive_label = "Layak Minum"
        negative_label = "Tidak Layak Minum"

    elif dataset_name == "cardio_train.csv":
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
    # INPUT DATA MANUAL
    # =========================
    st.subheader("Input Data Manual")

    st.write(
        "Masukkan nilai setiap fitur secara **manual** "
        "untuk memprediksi **data baru di luar dataset pelatihan**."
    )

    input_data = {}

    for col in feature_columns:

        # Default value = rata-rata (aman & masuk akal)
        if pd.api.types.is_numeric_dtype(df[col]):
            default_value = float(df[col].mean())
        else:
            default_value = 0.0

        input_data[col] = st.number_input(
            label=f"Nilai {col}",
            value=default_value,
            format="%.4f"
        )

    # Buat DataFrame dari input manual
    input_df = pd.DataFrame([input_data])

    st.markdown("**Data input yang digunakan untuk prediksi:**")
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
            "Hasil prediksi diperoleh dari **model terbaik** "
            "berdasarkan evaluasi **F1-Score** "
            "pada tahap Machine Learning."
        )

    # =========================
    # CATATAN AKADEMIK
    # =========================
    st.markdown("---")
    st.info(
        "Catatan:\n"
        "- Data untuk prediksi diinput secara **manual**.\n"
        "- Prediksi merepresentasikan proses **inferensi model**.\n"
        "- Hasil prediksi digunakan untuk **analisis dan pembelajaran**.\n"
        "- Model tidak dimaksudkan sebagai alat diagnosis medis."
    )
