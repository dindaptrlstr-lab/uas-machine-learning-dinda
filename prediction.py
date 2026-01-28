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
    menggunakan **model terbaik** hasil pelatihan pada menu
    **Machine Learning**.

    Data untuk prediksi **diinput secara manual** dan **bukan berasal
    dari dataset pelatihan**, sehingga merepresentasikan proses
    **inferensi model**.
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
    # LABEL HASIL PREDIKSI
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
    # INPUT DATA MANUAL (KE SAMPING)
    # =========================
    st.subheader("Input Data Manual")

    st.write(
        "Masukkan nilai setiap fitur secara **manual**. "
        "Input disusun **ke samping** agar lebih ringkas dan mudah dibaca."
    )

    data_input = {}

    # Buat kolom (3 input per baris)
    cols = st.columns(3)

    for i, kolom in enumerate(feature_columns):
        col = cols[i % 3]

        with col:
            if pd.api.types.is_numeric_dtype(df[kolom]):
                nilai_awal = float(df[kolom].mean())
            else:
                nilai_awal = 0.0

            data_input[kolom] = st.number_input(
                label=f"{kolom}",
                value=nilai_awal,
                format="%.2f"
            )

    # Bentuk DataFrame (tidak ditampilkan)
    input_df = pd.DataFrame([data_input])

    # =========================
    # PRA-PROSES DATA
    # =========================
    if scaler is not None:
        input_diproses = scaler.transform(input_df)
    else:
        input_diproses = input_df.values

    # =========================
    # PREDIKSI
    # =========================
    st.markdown("---")

    if st.button("🔍 Jalankan Prediksi"):

        hasil_prediksi = model.predict(input_diproses)[0]

        st.subheader("Hasil Prediksi")

        if hasil_prediksi == 1:
            st.success(f"✅ **{label_positif}**")
        else:
            st.error(f"❌ **{label_negatif}**")

        st.write(
            "Hasil prediksi diperoleh dari **model terbaik** "
            "berdasarkan evaluasi **F1-Score**."
        )

    # =========================
    # CATATAN
    # =========================
    st.markdown("---")
    st.info(
        "Catatan:\n"
        "- Data prediksi diinput secara **manual**.\n"
        "- Prediksi merepresentasikan proses **inferensi model**.\n"
        "- Hasil digunakan untuk **analisis dan pembelajaran**.\n"
        "- Bukan alat diagnosis medis."
    )
