import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib


def prediction_app():
    st.header("Prediksi Risiko Penyakit Kardiovaskular")
    st.write(
        "Masukkan data pasien untuk memprediksi risiko "
        "**penyakit kardiovaskular** menggunakan "
        "**model Machine Learning terbaik**."
    )

    st.divider()

    # ===============================
    # 1. Load Model & Metadata (SAFE)
    # ===============================
    try:
        model = joblib.load("best_model_cardio.pkl")
        feature_names = joblib.load("model_features.pkl")
        numeric_cols = joblib.load("numeric_columns.pkl")
    except Exception:
        st.error("❌ Model atau metadata belum tersedia.")
        st.stop()

    # ===============================
    # 2. Input Data Pasien
    # ===============================
    st.subheader("Input Data Pasien")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age = st.number_input("Age (hari)", min_value=0, max_value=40000, value=18000)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col3:
        height = st.number_input("Height (cm)", min_value=100, max_value=220, value=165)
    with col4:
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=65)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        ap_hi = st.number_input("Systolic BP", min_value=80, max_value=250, value=120)
    with col2:
        ap_lo = st.number_input("Diastolic BP", min_value=50, max_value=150, value=80)
    with col3:
        cholesterol = st.selectbox("Cholesterol", [1, 2, 3])
    with col4:
        gluc = st.selectbox("Glucose", [1, 2, 3])

    col1, col2, col3 = st.columns(3)
    with col1:
        smoke = st.selectbox("Smoking", [0, 1])
    with col2:
        alco = st.selectbox("Alcohol Intake", [0, 1])
    with col3:
        active = st.selectbox("Physical Activity", [0, 1])

    # ===============================
    # 3. Feature Engineering
    # ===============================
    bmi = weight / ((height / 100) ** 2)
    st.info(f"BMI Pasien: **{bmi:.2f}**")

    user_df = pd.DataFrame({
        "age": [age],
        "gender": [1 if gender == "Male" else 2],
        "height": [height],
        "weight": [weight],
        "ap_hi": [ap_hi],
        "ap_lo": [ap_lo],
        "cholesterol": [cholesterol],
        "gluc": [gluc],
        "smoke": [smoke],
        "alco": [alco],
        "active": [active],
        "bmi": [bmi]
    })

    # ===============================
    # 4. Encoding (SAFE)
    # ===============================
    user_processed = pd.get_dummies(user_df, drop_first=True)

    for col in feature_names:
        if col not in user_processed.columns:
            user_processed[col] = 0

    user_processed = user_processed[feature_names]

    # ===============================
    # 5. Normalisasi (KONSISTEN)
    # ===============================
    scaler = MinMaxScaler()
    user_processed[numeric_cols] = scaler.fit_transform(
        user_processed[numeric_cols]
    )

    # ===============================
    # 6. Prediction
    # ===============================
    if st.button("Prediksi Risiko"):
        prob = model.predict_proba(user_processed)[0][1]
        pred = model.predict(user_processed)[0]

        st.subheader("Hasil Prediksi")
        st.metric("Probabilitas Risiko", f"{prob * 100:.2f}%")

        if pred == 1:
            st.error("⚠️ Pasien **BERISIKO** penyakit kardiovaskular")
        else:
            st.success("✅ Pasien **TIDAK BERISIKO TINGGI** penyakit kardiovaskular")

        st.divider()

        st.markdown("### Interpretasi")
        st.markdown("""
        - Probabilitas tinggi menunjukkan potensi risiko kardiovaskular
        - Model digunakan sebagai **alat skrining awal**
        - Keputusan medis tetap harus melibatkan **tenaga kesehatan**
        """)


if __name__ == "__main__":
    prediction_app()
