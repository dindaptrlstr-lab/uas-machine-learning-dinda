import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib


def prediction_app():
    st.title("Prediksi Risiko Penyakit Kardiovaskular")
    st.write(
        "Masukkan data pasien untuk memprediksi risiko **penyakit kardiovaskular** "
        "menggunakan **model Machine Learning terbaik**."
    )

    # ===============================
    # 1. Load Model & Metadata
    # ===============================
    model = joblib.load("best_model_cardio.pkl")
    feature_names = joblib.load("model_features.pkl")
    numeric_cols = joblib.load("numeric_columns.pkl")

    # ===============================
    # 2. Input User (SESUAI DATASET)
    # ===============================
    st.subheader("Input Data Pasien")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age = st.number_input("Age (hari)", 0, 40000, 18000)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col3:
        height = st.number_input("Height (cm)", 100, 220, 165)
    with col4:
        weight = st.number_input("Weight (kg)", 30, 200, 65)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        ap_hi = st.number_input("Systolic BP", 80, 250, 120)
    with col2:
        ap_lo = st.number_input("Diastolic BP", 50, 150, 80)
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
    st.info(f"📊 BMI Pasien: **{bmi:.2f}**")

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
    # 4. Encoding
    # ===============================
    user_processed = pd.get_dummies(user_df, drop_first=True)

    for col in feature_names:
        if col not in user_processed.columns:
            user_processed[col] = 0

    user_processed = user_processed[feature_names]

    # ===============================
    # 5. Normalisasi
    # ===============================
    scaler = MinMaxScaler()
    user_processed[numeric_cols] = scaler.fit_transform(user_processed[numeric_cols])

    # ===============================
    # 6. Prediction
    # ===============================
    if st.button("Prediksi Risiko"):
        prob = model.predict_proba(user_processed)[0][1]
        pred = model.predict(user_processed)[0]

        st.subheader("Hasil Prediksi")
        st.metric("Probabilitas Risiko", f"{prob*100:.2f}%")

        if pred == 1:
            st.error("⚠️ Pasien **BERISIKO** penyakit kardiovaskular")
        else:
            st.success("✅ Pasien **TIDAK BERISIKO TINGGI** penyakit kardiovaskular")

        st.write("---")
        st.write("### Interpretasi")
        st.markdown("""
        - Probabilitas tinggi menunjukkan pasien berpotensi mengalami penyakit kardiovaskular
        - Model digunakan sebagai **alat skrining awal**
        - Keputusan medis tetap harus melibatkan tenaga kesehatan
        """)


if __name__ == "__main__":
    prediction_app()
