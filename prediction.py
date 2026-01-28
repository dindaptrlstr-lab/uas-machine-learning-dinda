import streamlit as st
import pandas as pd
import numpy as np


def prediction_page():
    st.subheader("Aplikasi Prediksi Kelayakan Air Minum dan Risiko Penyakit Jantung")

    # =========================
    # DESKRIPSI HALAMAN
    # =========================
    st.markdown("""
    Halaman ini digunakan untuk melakukan **prediksi pada data baru**
    menggunakan **model terbaik** yang diperoleh dari proses pelatihan
    pada menu **Pemodelan Machine Learning**.

    Data yang dimasukkan bersifat **manual** dan tidak berasal dari dataset pelatihan,
    sehingga merepresentasikan proses **inferensi model**.
    """)

    # =========================
    # PENGAMAN ALUR PIPELINE
    # =========================
    if "best_model" not in st.session_state:
        st.warning("Silakan jalankan menu **Pemodelan Machine Learning** terlebih dahulu.")
        return

    if "df" not in st.session_state or "dataset_name" not in st.session_state:
        st.warning("Silakan unggah dataset terlebih dahulu.")
        return

    if "feature_columns" not in st.session_state:
        st.warning("Informasi fitur tidak tersedia. Silakan lakukan pelatihan ulang.")
        return

    # =========================
    # AMBIL OBJEK SESSION
    # =========================
    model = st.session_state["best_model"]
    df = st.session_state["df"]
    dataset_name = st.session_state["dataset_name"]
    feature_columns = st.session_state["feature_columns"]
    scaler = st.session_state.get("scaler")

    feature_columns = [f for f in feature_columns if f.lower() != "id"]

    # =========================
    # LABEL HASIL
    # =========================
    if dataset_name == "cardio_train.csv":
        label_positif = "Berisiko Penyakit Jantung"
        label_negatif = "Tidak Berisiko Penyakit Jantung"
        tipe_data = "kesehatan"
    elif dataset_name == "water_potability.csv":
        label_positif = "Air Layak Minum"
        label_negatif = "Air Tidak Layak Minum"
        tipe_data = "air"
    else:
        st.error("Dataset tidak dikenali.")
        return

    # =========================
    # INPUT DATA
    # =========================
    st.markdown("---")
    st.subheader("Input Data")
    st.write("Silakan masukkan nilai fitur berikut untuk melakukan prediksi.")

    data_input = {}
    cols = st.columns(3)

    for i, kolom in enumerate(feature_columns):
        col = cols[i % 3]
        with col:
            nilai_awal = float(df[kolom].mean())
            data_input[kolom] = st.number_input(
                label=kolom,
                value=nilai_awal,
                format="%.2f"
            )

    # =========================
    # PRA-PEMROSESAN
    # =========================
    input_df = pd.DataFrame([data_input])
    input_df = input_df[feature_columns]

    if scaler is not None:
        input_diproses = scaler.transform(input_df)
    else:
        input_diproses = input_df.values

    # =========================
    # PREDIKSI + REKOMENDASI
    # =========================
    st.markdown("---")
    if st.button("🔍 Jalankan Prediksi"):

        hasil_prediksi = model.predict(input_diproses)[0]

        st.subheader("Hasil Prediksi")

        # =====================
        # HASIL + REKOMENDASI AIR
        # =====================
        if tipe_data == "air":
            if hasil_prediksi == 1:
                st.success("✅ **Air Layak Minum**")

                st.markdown("### ✅ Rekomendasi")
                st.markdown("""
                Berdasarkan hasil prediksi, kualitas air **layak untuk dikonsumsi**.

                **Saran:**
                - Air dapat digunakan untuk kebutuhan sehari-hari
                - Tetap lakukan pengecekan kualitas air secara berkala
                - Simpan air di wadah yang bersih dan tertutup
                """)
            else:
                st.error("❌ **Air Tidak Layak Minum**")

                st.markdown("### ⚠️ Rekomendasi")
                st.markdown("""
                Berdasarkan hasil prediksi, air **tidak layak untuk dikonsumsi langsung**.

                **Saran:**
                - Lakukan penyaringan atau perebusan sebelum digunakan
                - Gunakan alat filtrasi air (filter / RO)
                - Hindari konsumsi langsung tanpa pengolahan
                """)

        # =====================
        # HASIL + REKOMENDASI KESEHATAN
        # =====================
        if tipe_data == "kesehatan":
            if hasil_prediksi == 1:
                st.error("⚠️ **Berisiko Penyakit Jantung**")

                st.markdown("### ⚠️ Rekomendasi")
                st.markdown("""
                Berdasarkan hasil prediksi, terdapat **risiko penyakit jantung**.

                **Saran:**
                - Lakukan pemeriksaan kesehatan secara berkala
                - Jaga pola makan sehat dan rendah lemak
                - Tingkatkan aktivitas fisik
                - Konsultasi dengan tenaga medis
                """)
            else:
                st.success("✅ **Tidak Berisiko Penyakit Jantung**")

                st.markdown("### ✅ Rekomendasi")
                st.markdown("""
                Berdasarkan hasil prediksi, risiko penyakit jantung **tergolong rendah**.

                **Saran:**
                - Pertahankan pola hidup sehat
                - Tetap aktif berolahraga
                - Lakukan pemeriksaan kesehatan rutin
                """)

    # =========================
    # CATATAN
    # =========================
    st.markdown("---")
    st.info(
        "Catatan:\n"
        "- Hasil prediksi merupakan **hasil inferensi model**, "
        "bukan diagnosis medis dan tidak menggantikan pemeriksaan profesional."
    )
