import streamlit as st
import pandas as pd
import numpy as np


def prediction_page():
    st.subheader("Aplikasi Prediksi Kelayakan Air Minum dan Risiko Penyakit Jantung")

    # =========================
    # AMBIL NAMA MODEL TERBAIK
    # =========================
    nama_model = st.session_state.get("best_model_name", "model terpilih")

    # =========================
    # DESKRIPSI HALAMAN
    # =========================
    st.markdown(f"""
    Halaman ini digunakan untuk melakukan **prediksi pada data baru**
    menggunakan **model {nama_model}** yang memiliki kinerja terbaik
    berdasarkan hasil evaluasi pada tahap pemodelan.

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

    # Hilangkan kolom ID jika ada
    feature_columns = [f for f in feature_columns if f.lower() != "id"]

    # =========================
    # PENENTUAN TIPE & LABEL
    # =========================
    if dataset_name == "water_potability.csv":
        tipe_data = "air"
        label_positif = "Air Layak Minum"
        label_negatif = "Air Tidak Layak Minum"
    elif dataset_name == "cardio_train.csv":
        tipe_data = "kesehatan"
        label_positif = "Berisiko Penyakit Jantung"
        label_negatif = "Tidak Berisiko Penyakit Jantung"
    else:
        st.error("Dataset tidak dikenali.")
        return

    # =========================
    # FORM INPUT DATA
    # =========================
    st.markdown("---")
    st.subheader("Input Data")
    st.write("Silakan masukkan nilai parameter berikut untuk melakukan prediksi.")

    data_input = {}
    cols = st.columns(3)

    for i, kolom in enumerate(feature_columns):
        with cols[i % 3]:
            nilai_awal = float(df[kolom].mean())
            data_input[kolom] = st.number_input(
                label=kolom,
                value=nilai_awal,
                format="%.2f"
            )

    # =========================
    # PRA-PEMROSESAN DATA INPUT
    # =========================
    input_df = pd.DataFrame([data_input])

    # WAJIB: samakan urutan kolom dengan training
    input_df = input_df[feature_columns]

    if scaler is not None:
        input_diproses = scaler.transform(input_df)
    else:
        input_diproses = input_df.values

    # =========================
    # PROSES PREDIKSI + REKOMENDASI
    # =========================
    st.markdown("---")
    if st.button("🔍 Jalankan Prediksi"):

        hasil = model.predict(input_diproses)[0]

        st.subheader("Hasil Prediksi")

        # ===== DATASET AIR =====
        if tipe_data == "air":
            if hasil == 1:
                st.success("✅ **Air Layak Minum**")
                st.markdown("### Rekomendasi")
                st.markdown("""
                Berdasarkan hasil prediksi, kualitas air **layak untuk dikonsumsi**.

                **Saran:**
                - Air dapat digunakan untuk kebutuhan sehari-hari
                - Lakukan pengecekan kualitas air secara berkala
                - Simpan air di wadah yang bersih dan tertutup
                """)
            else:
                st.error("❌ **Air Tidak Layak Minum**")
                st.markdown("### ⚠️ Rekomendasi")
                st.markdown("""
                Berdasarkan hasil prediksi, air **tidak layak untuk dikonsumsi langsung**.

                **Saran:**
                - Lakukan perebusan atau penyaringan air
                - Gunakan alat filtrasi air
                - Hindari konsumsi langsung tanpa pengolahan
                """)

        # ===== DATASET KESEHATAN =====
        if tipe_data == "kesehatan":
            if hasil == 1:
                st.error("⚠️ **Berisiko Penyakit Jantung**")
                st.markdown("### ⚠️ Rekomendasi")
                st.markdown("""
                Berdasarkan hasil prediksi, terdapat **risiko penyakit jantung**.

                **Saran:**
                - Lakukan pemeriksaan kesehatan secara rutin
                - Terapkan pola makan sehat
                - Tingkatkan aktivitas fisik
                - Konsultasikan dengan tenaga medis
                """)
            else:
                st.success("✅ **Tidak Berisiko Penyakit Jantung**")
                st.markdown("### ✅ Rekomendasi")
                st.markdown("""
                Berdasarkan hasil prediksi, risiko penyakit jantung **tergolong rendah**.

                **Saran:**
                - Pertahankan pola hidup sehat
                - Tetap aktif berolahraga
                - Lakukan pemeriksaan kesehatan berkala
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
