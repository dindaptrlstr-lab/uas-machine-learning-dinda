import streamlit as st
import pandas as pd
import numpy as np

def prediction_page():
    st.title("🔮 Prediction App")

    # =========================
    # PENGAMAN
    # =========================
    if "best_model" not in st.session_state:
        st.warning("Silakan jalankan menu Machine Learning terlebih dahulu.")
        return

    if "df" not in st.session_state or "dataset_name" not in st.session_state:
        st.warning("Silakan upload dataset terlebih dahulu.")
        return

    model = st.session_state["best_model"]
    df = st.session_state["df"]
    dataset_name = st.session_state["dataset_name"]

    scaler = st.session_state.get("scaler")
    feature_columns = st.session_state.get("feature_columns")

    # =========================
    # TARGET OTOMATIS
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
        "Menu ini melakukan **prediksi otomatis** menggunakan "
        "**model terbaik** hasil training sebelumnya."
    )

    st.markdown("---")

    # =========================
    # INPUT OTOMATIS
    # =========================
    st.subheader("📥 Data Input (Otomatis)")

    input_df = df[feature_columns].iloc[-1:].copy()

    st.write(
        "Data diambil dari **baris terakhir dataset** "
        "dan digunakan sebagai input prediksi."
    )
    st.dataframe(input_df, use_container_width=True)

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
        st.subheader("📊 Hasil Prediksi")

        if prediction == 1:
            st.success(f"✅ **{positive_label}**")
        else:
            st.error(f"❌ **{negative_label}**")

        st.write(
            "Prediksi ini dihasilkan menggunakan **model terbaik** "
            "berdasarkan evaluasi F1-Score pada menu Machine Learning."
        )

    st.markdown("---")
    st.info(
        "Catatan:\n"
        "- Prediksi bersifat **klasifikasi**, bukan prediksi waktu.\n"
        "- Data input berasal dari dataset yang di-upload.\n"
        "- Hasil prediksi hanya untuk tujuan **analisis dan pembelajaran**."
    )
