import streamlit as st
import pandas as pd
import numpy as np


def prediction_page():
    st.subheader("Sistem Prediksi Kelayakan Air Minum serta Risiko Penyakit Kardiovaskular")

    # =========================
    # DESKRIPSI HALAMAN
    # =========================
    st.markdown("""
    Halaman ini digunakan untuk memasukkan data pengguna 
    dan memperoleh hasil prediksi dari model Machine Learning.
    """)
    st.markdown("---")

    # =========================
    # PENGAMAN PIPELINE
    # =========================
    if "best_model" not in st.session_state:
        st.warning("Silakan jalankan menu **Machine Learning** terlebih dahulu.")
        return

    if "df" not in st.session_state or "dataset_name" not in st.session_state:
        st.warning("Silakan upload dataset terlebih dahulu.")
        return

    if "feature_columns" not in st.session_state:
        st.warning("Informasi fitur tidak tersedia. Silakan lakukan training ulang.")
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
    # LABEL TARGET
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

    # =========================
    # MAPPING LABEL (BAHASA INDONESIA)
    # =========================
    label_mapping = {
        "age": "Umur (hari)",
        "gender": "Jenis Kelamin (1 = Pria, 2 = Wanita)",
        "height": "Tinggi Badan (cm)",
        "weight": "Berat Badan (kg)",
        "ap_hi": "Tekanan Darah Sistolik",
        "ap_lo": "Tekanan Darah Diastolik",
        "cholesterol": "Kolesterol (1=Normal, 2=Tinggi, 3=Sangat Tinggi)",
        "gluc": "Gula Darah (1=Normal, 2=Tinggi, 3=Sangat Tinggi)",
        "smoke": "Merokok (0=Tidak, 1=Ya)",
        "alco": "Konsumsi Alkohol (0=Tidak, 1=Ya)",
        "active": "Aktivitas Fisik (0=Tidak Aktif, 1=Aktif)"
    }

    # =========================
    # INPUT MANUAL DATA
    # =========================
    st.subheader("Input Data")

    input_data = {}
    cols = st.columns(2)

    # HILANGKAN ID DARI INPUT
    filtered_features = [f for f in feature_columns if f != "id"]

    for i, feature in enumerate(filtered_features):
        col = cols[i % 2]

        label_ui = label_mapping.get(feature, feature)

        input_data[feature] = col.number_input(
            label=label_ui,
            min_value=int(df[feature].min()),
            max_value=int(df[feature].max()),
            value=int(df[feature].mean()),
            step=1,
            format="%d"
        )

    input_df = pd.DataFrame([input_data])

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
        st.subheader("📌 Hasil Prediksi")

        if prediction == 1:
            st.success(f"✅ **{positive_label}**")
        else:
            st.error(f"❌ **{negative_label}**")

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
        "- Data diinput secara manual oleh pengguna.\n"
        "- Model bersifat **klasifikasi**, bukan diagnosis medis.\n"
        "- Digunakan untuk **pembelajaran dan analisis data**."
    )
