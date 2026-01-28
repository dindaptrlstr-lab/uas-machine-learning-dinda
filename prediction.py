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

    Data dimasukkan secara **manual**, tidak berasal dari dataset pelatihan,
    sehingga mencerminkan proses **inferensi model**.
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
    # HAPUS KOLOM ID (TIDAK DIPAKAI)
    # =========================
    feature_columns = [f for f in feature_columns if f.lower() != "id"]

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
    # LABEL INPUT BAHASA INDONESIA
    # =========================
    label_indonesia = {
        "age": "Usia (hari)",
        "gender": "Jenis Kelamin (1 = Pria, 2 = Wanita)",
        "height": "Tinggi Badan (cm)",
        "weight": "Berat Badan (kg)",
        "ap_hi": "Tekanan Darah Sistolik",
        "ap_lo": "Tekanan Darah Diastolik",
        "cholesterol": "Kadar Kolesterol",
        "gluc": "Kadar Glukosa",
        "smoke": "Kebiasaan Merokok (0 = Tidak, 1 = Ya)",
        "alco": "Konsumsi Alkohol (0 = Tidak, 1 = Ya)",
        "active": "Aktivitas Fisik (0 = Tidak, 1 = Ya)"
    }

    # =========================
    # INPUT DATA MANUAL (KE SAMPING)
    # =========================
    st.subheader("Input Data Manual")

    st.write(
        "Masukkan nilai setiap variabel berikut "
        "untuk melakukan prediksi data baru."
    )

    data_input = {}
    cols = st.columns(3)

    for i, kolom in enumerate(feature_columns):
        col = cols[i % 3]

        with col:
            nilai_awal = float(df[kolom].mean())

            label_tampil = label_indonesia.get(kolom, kolom)

            data_input[kolom] = st.number_input(
                label=label_tampil,
                value=nilai_awal,
                format="%.2f"
            )

    input_df = pd.DataFrame([data_input])

    # =========================
    # PRA-PROSES DATA
    # =========================
    if scaler is not None:
        input_diproses = scaler.transform(input_df)
    else:
        input_diproses = input_df.values

    # =========================
    # JALANKAN PREDIKSI
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
        "- Kolom **ID tidak digunakan** karena tidak berpengaruh terhadap prediksi.\n"
        "- Data dimasukkan secara **manual**.\n"
        "- Prediksi merepresentasikan proses **inferensi model**.\n"
        "- Bukan alat diagnosis medis."
    )
