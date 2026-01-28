import streamlit as st
import pandas as pd
import numpy as np


def prediction_page():
    st.subheader("Aplikasi Prediksi")

    # =========================
    # DESKRIPSI HALAMAN
    # =========================
    st.markdown("""
    Halaman ini digunakan untuk melakukan **prediksi data baru**
    menggunakan **model terbaik** yang diperoleh dari proses pelatihan
    pada menu **Machine Learning**.

    Data yang digunakan untuk prediksi **diinput secara manual**
    dan **tidak berasal dari dataset pelatihan**, sehingga mencerminkan
    proses **inferensi model**.
    """)
    st.markdown("---")

    # =========================
    # PENGAMAN PIPELINE
    # =========================
    if "best_model" not in st.session_state:
        st.warning("Silakan jalankan menu **Machine Learning** terlebih dahulu.")
        return

    if "df" not in st.session_state or "dataset_name" not in st.session_state:
        st.warning("Silakan unggah dataset terlebih dahulu melalui sidebar.")
        return

    if "feature_columns" not in st.session_state:
        st.warning("Informasi fitur tidak tersedia. Silakan lakukan pelatihan ulang.")
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
    # LABEL TARGET OTOMATIS
    # =========================
    if dataset_name == "water_potability.csv":
        label_positif = "Layak Minum"
        label_negatif = "Tidak Layak Minum"

    elif dataset_name == "cardio_train.csv":
        label_positif = "Berisiko Penyakit Jantung"
        label_negatif = "Tidak Berisiko"

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
        "Masukkan nilai setiap fitur secara **manual** untuk melakukan "
        "prediksi pada **data baru di luar dataset pelatihan**."
    )

    data_input = {}

    for kolom in feature_columns:

        # Nilai awal menggunakan rata-rata agar memudahkan pengguna
        if pd.api.types.is_numeric_dtype(df[kolom]):
            nilai_awal = float(df[kolom].mean())
        else:
            nilai_awal = 0.0

        data_input[kolom] = st.number_input(
            label=f"Nilai {kolom}",
            value=nilai_awal,
            format="%.4f"
        )

    # Membentuk DataFrame (tanpa ditampilkan)
    input_df = pd.DataFrame([data_input])

    # =========================
    # PRA-PROSES DATA (KONSISTEN)
    # =========================
    if scaler is not None:
        input_diproses = scaler.transform(input_df)
    else:
        input_diproses = input_df.values

    # =========================
    # JALANKAN PREDIKSI
    # =========================
    if st.button("🔍 Jalankan Prediksi"):

        hasil_prediksi = model.predict(input_diproses)[0]

        st.markdown("---")
        st.subheader("Hasil Prediksi")

        if hasil_prediksi == 1:
            st.success(f"✅ **{label_positif}**")
        else:
            st.error(f"❌ **{label_negatif}**")

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
