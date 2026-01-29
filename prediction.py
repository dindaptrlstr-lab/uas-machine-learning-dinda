import streamlit as st
import pandas as pd
import numpy as np


def prediction_page():
    st.subheader("Sistem Prediksi Kelayakan Air Minum dan Risiko Penyakit Jantung")
    st.markdown(
    "Sistem ini dirancang untuk membantu proses analisis dan pembelajaran Machine Learning "
    "dalam memprediksi kelayakan air minum serta risiko penyakit jantung berdasarkan "
    "parameter input yang diberikan oleh pengguna."
)
    st.markdown("---")

    # =========================
    # PENGAMAN PIPELINE
    # =========================
    if "best_model" not in st.session_state:
        st.warning("Silakan jalankan menu **Machine Learning** terlebih dahulu.")
        return

    if "feature_columns" not in st.session_state:
        st.warning("Informasi fitur tidak tersedia. Silakan training ulang.")
        return

    if "dataset_name" not in st.session_state:
        st.warning("Dataset belum dikenali.")
        return

    # =========================
    # AMBIL SESSION STATE
    # =========================
    model = st.session_state["best_model"]
    scaler = st.session_state.get("scaler", None)
    feature_columns = st.session_state["feature_columns"]
    dataset_name = st.session_state["dataset_name"]

    # =========================
    # INPUT DATA
    # =========================
    st.subheader("Input Data")

    # =====================================================
    # DATASET 1 : CARDIOVASCULAR
    # =====================================================
    if dataset_name == "cardio_train.csv":

        col1, col2 = st.columns(2)

        age_years = col1.number_input("Umur (tahun)", 10, 100, 50, step=1)
        age = age_years * 365

        gender = 1 if col2.selectbox("Jenis Kelamin", ["Pria", "Wanita"]) == "Pria" else 2

        height = col1.number_input("Tinggi Badan (cm)", 100, 220, 165, step=1)
        weight = col2.number_input("Berat Badan (kg)", 30, 200, 70, step=1)

        ap_hi = col1.number_input("Tekanan Darah Sistolik", 80, 250, 120, step=1)
        ap_lo = col2.number_input("Tekanan Darah Diastolik", 50, 150, 80, step=1)

        cholesterol = {"Normal": 1, "Tinggi": 2, "Sangat Tinggi": 3}[
            col1.selectbox("Kolesterol", ["Normal", "Tinggi", "Sangat Tinggi"])
        ]

        gluc = {"Normal": 1, "Tinggi": 2, "Sangat Tinggi": 3}[
            col2.selectbox("Gula Darah", ["Normal", "Tinggi", "Sangat Tinggi"])
        ]

        smoke = 1 if col1.selectbox("Merokok", ["Tidak", "Ya"]) == "Ya" else 0
        alco = 1 if col2.selectbox("Konsumsi Alkohol", ["Tidak", "Ya"]) == "Ya" else 0
        active = 1 if col2.selectbox("Aktivitas Fisik", ["Tidak Aktif", "Aktif"]) == "Aktif" else 0

        input_df = pd.DataFrame([{
            "age": int(age),
            "gender": int(gender),
            "height": int(height),
            "weight": int(weight),
            "ap_hi": int(ap_hi),
            "ap_lo": int(ap_lo),
            "cholesterol": int(cholesterol),
            "gluc": int(gluc),
            "smoke": int(smoke),
            "alco": int(alco),
            "active": int(active)
        }])

        positive_label = "Tidak Berisiko Penyakit Jantung"
        negative_label = "Berisiko Penyakit Jantung"

    # =====================================================
    # DATASET 2 : WATER POTABILITY
    # =====================================================
    elif dataset_name == "water_potability.csv":

        col1, col2 = st.columns(2)

        input_df = pd.DataFrame([{
            "ph": int(col1.number_input("ph", 0, 14, 7, step=1)),
            "Hardness": int(col2.number_input("Hardness", 0, 500, 200, step=1)),
            "Solids": int(col1.number_input("Solids", 0, 50000, 20000, step=1)),
            "Chloramines": int(col2.number_input("Chloramines", 0, 20, 7, step=1)),
            "Sulfate": int(col1.number_input("Sulfate", 0, 500, 330, step=1)),
            "Conductivity": int(col2.number_input("Conductivity", 0, 1000, 420, step=1)),
            "Organic_carbon": int(col1.number_input("Organic_carbon", 0, 50, 14, step=1)),
            "Trihalomethanes": int(col2.number_input("Trihalomethanes", 0, 200, 66, step=1)),
            "Turbidity": int(col1.number_input("Turbidity", 0, 20, 4, step=1))
        }])

        positive_label = "Air Layak Minum"
        negative_label = "Air Tidak Layak Minum"

    else:
        st.error("Dataset tidak dikenali.")
        return

    # =========================
    # PENYESUAIAN FITUR
    # =========================
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_columns]

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
        pred = model.predict(input_processed)[0]

        st.markdown("---")
        st.subheader("Hasil Prediksi")

        if pred == 1:
            st.success(f"✅ **{positive_label}**")
        else:
            st.error(f"⚠️ **{negative_label}**")

    # =========================
    # CATATAN
    # =========================
    st.markdown("---")
    st.info(
        "Catatan:\n"
        "- Sistem ini digunakan untuk **pembelajaran Machine Learning**.\n"
        "- Hasil prediksi **bukan diagnosis atau uji laboratorium**."
    )



