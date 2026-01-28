import streamlit as st
import pandas as pd
import numpy as np


def prediction_page():
    st.subheader("Aplikasi Prediksi Kelayakan Air Minum dan Risiko Penyakit Jantung")

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
    # AMBIL OBJEK SESSION
    # =========================
    model = st.session_state["best_model"]
    df = st.session_state["df"]
    dataset_name = st.session_state["dataset_name"]
    feature_columns = st.session_state["feature_columns"]
    scaler = st.session_state.get("scaler")

    # Hapus kolom ID
    feature_columns = [f for f in feature_columns if f.lower() != "id"]

    # =========================
    # LABEL HASIL PREDIKSI
    # =========================
    if dataset_name == "cardio_train.csv":
        label_positif = "Berisiko Penyakit Jantung"
        label_negatif = "Tidak Berisiko"
    elif dataset_name == "water_potability.csv":
        label_positif = "Layak Minum"
        label_negatif = "Tidak Layak Minum"
    else:
        st.error("Dataset tidak dikenali.")
        return

    st.markdown("---")
    st.subheader("Input Data")
    st.write("Masukkan data berikut untuk melakukan prediksi.")

    # =========================
    # INPUT DATA (KE SAMPING)
    # =========================
    data_input = {}
    cols = st.columns(3)

    for i, kolom in enumerate(feature_columns):
        col = cols[i % 3]

        with col:
            # ===== JENIS KELAMIN =====
            if kolom == "gender":
                pilihan = st.selectbox(
                    "Jenis Kelamin",
                    ["Pria", "Wanita"]
                )
                # Mapping sesuai dataset cardio
                data_input[kolom] = 2 if pilihan == "Pria" else 1

            # ===== PERILAKU (YA / TIDAK) =====
            elif kolom == "smoke":
                pilihan = st.selectbox("Kebiasaan Merokok", ["Tidak", "Ya"])
                data_input[kolom] = 1 if pilihan == "Ya" else 0

            elif kolom == "alco":
                pilihan = st.selectbox("Konsumsi Alkohol", ["Tidak", "Ya"])
                data_input[kolom] = 1 if pilihan == "Ya" else 0

            elif kolom == "active":
                pilihan = st.selectbox("Aktif Berolahraga", ["Tidak", "Ya"])
                data_input[kolom] = 1 if pilihan == "Ya" else 0

            # ===== NUMERIK =====
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

        hasil = model.predict(input_diproses)[0]

        st.subheader("Hasil Prediksi")
        if hasil == 1:
            st.success(f"✅ **{label_positif}**")
        else:
            st.error(f"❌ **{label_negatif}**")

    # =========================
    # CATATAN
    # =========================
    st.markdown("---")
    st.info(
        "Catatan:\n"
     "- Hasil prediksi merupakan **inferensi model**, bukan diagnosis medis."
    )

