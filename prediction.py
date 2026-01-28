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
        st.warning(
            "Silakan jalankan proses **Pemodelan Machine Learning** "
            "terlebih dahulu."
        )
        return

    if "df" not in st.session_state or "dataset_name" not in st.session_state:
        st.warning(
            "Silakan unggah dataset terlebih dahulu "
            "melalui menu Pemilihan Dataset."
        )
        return

    if "feature_columns" not in st.session_state:
        st.warning(
            "Informasi fitur tidak tersedia. "
            "Silakan lakukan pelatihan ulang model."
        )
        return

    # =========================
    # AMBIL OBJEK DARI SESSION
    # =========================
    model = st.session_state["best_model"]
    df = st.session_state["df"]
    dataset_name = st.session_state["dataset_name"]
    feature_columns = st.session_state["feature_columns"]
    scaler = st.session_state.get("scaler")

    # Hilangkan kolom ID jika ada
    feature_columns = [f for f in feature_columns if f.lower() != "id"]

    # =========================
    # PENENTUAN LABEL HASIL
    # =========================
    if dataset_name == "cardio_train.csv":
        label_positif = "Berisiko Penyakit Jantung"
        label_negatif = "Tidak Berisiko Penyakit Jantung"
    elif dataset_name == "water_potability.csv":
        label_positif = "Air Layak Minum"
        label_negatif = "Air Tidak Layak Minum"
    else:
        st.error("Dataset tidak dikenali oleh sistem.")
        return

    # =========================
    # FORM INPUT DATA
    # =========================
    st.markdown("---")
    st.subheader("Input Data")
    st.write("Silakan masukkan nilai fitur berikut untuk melakukan prediksi.")

    data_input = {}
    cols = st.columns(3)

    for i, kolom in enumerate(feature_columns):
        col = cols[i % 3]

        with col:
            # ===== JENIS KELAMIN =====
            if kolom == "gender":
                pilihan = st.selectbox("Jenis Kelamin", ["Pria", "Wanita"])
                data_input[kolom] = 2 if pilihan == "Pria" else 1

            # ===== PERILAKU (YA / TIDAK) =====
            elif kolom == "smoke":
                pilihan = st.selectbox("Kebiasaan Merokok", ["Tidak", "Ya"])
                data_input[kolom] = 1 if pilihan == "Ya" else 0

            elif kolom == "alco":
                pilihan = st.selectbox("Konsumsi Alkohol", ["Tidak", "Ya"])
                data_input[kolom] = 1 if pilihan == "Ya" else 0

            elif kolom == "active":
                pilihan = st.selectbox("Aktivitas Fisik", ["Tidak Aktif", "Aktif"])
                data_input[kolom] = 1 if pilihan == "Aktif" else 0

            # ===== FITUR NUMERIK =====
            else:
                nilai_awal = float(df[kolom].mean())

                label_indonesia = {
                    "age": "Usia",
                    "height": "Tinggi Badan (cm)",
                    "weight": "Berat Badan (kg)",
                    "ap_hi": "Tekanan Darah Sistolik",
                    "ap_lo": "Tekanan Darah Diastolik",
                    "cholesterol": "Kadar Kolesterol",
                    "gluc": "Kadar Glukosa"
                }

                label_tampil = label_indonesia.get(kolom, kolom)

                data_input[kolom] = st.number_input(
                    label=label_tampil,
                    value=nilai_awal,
                    format="%.2f"
                )

    # =========================
    # DATAFRAME INPUT (FIX ERROR)
    # =========================
    input_df = pd.DataFrame([data_input])

    # ⚠️ PENTING: SAMAKAN URUTAN KOLOM DENGAN SAAT TRAINING
    input_df = input_df[feature_columns]

    # =========================
    # PRA-PEMROSESAN DATA INPUT
    # =========================
    if scaler is not None:
        input_diproses = scaler.transform(input_df)
    else:
        input_diproses = input_df.values

    # =========================
    # PROSES PREDIKSI
    # =========================
    st.markdown("---")
    if st.button("🔍 Jalankan Prediksi"):

        hasil_prediksi = model.predict(input_diproses)[0]

        st.subheader("Hasil Prediksi")
        if hasil_prediksi == 1:
            st.success(f"✅ **{label_positif}**")
        else:
            st.error(f"❌ **{label_negatif}**")

    # =========================
    # CATATAN PENTING
    # =========================
    st.markdown("---")
    st.info(
        "Catatan:\n"
        "- Hasil prediksi merupakan **hasil inferensi model**, "
        "bukan diagnosis medis dan tidak menggantikan "
        "pemeriksaan oleh tenaga kesehatan profesional."
    )
