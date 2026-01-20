import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib


def prediction_app():
    st.header("Prediksi Kelayakan Air Minum")
    st.write(
        "Masukkan parameter kualitas air untuk memprediksi "
        "**kelayakan air minum (Potability)** menggunakan "
        "**model Machine Learning terbaik**."
    )

    st.divider()

    # ===============================
    # 1. Load Model & Metadata
    # ===============================
    try:
        model = joblib.load("best_model_water.pkl")
        feature_names = joblib.load("model_features.pkl")
        numeric_cols = joblib.load("numeric_columns.pkl")
    except Exception:
        st.error("❌ Model atau metadata belum tersedia.")
        st.stop()

    # ===============================
    # 2. Input Parameter Air
    # ===============================
    st.subheader("Input Parameter Kualitas Air")

    col1, col2, col3 = st.columns(3)
    with col1:
        ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
    with col2:
        hardness = st.number_input("Hardness", min_value=0.0, value=200.0)
    with col3:
        solids = st.number_input("Total Dissolved Solids", min_value=0.0, value=20000.0)

    col1, col2, col3 = st.columns(3)
    with col1:
        chloramines = st.number_input("Chloramines", min_value=0.0, value=7.0)
    with col2:
        sulfate = st.number_input("Sulfate", min_value=0.0, value=300.0)
    with col3:
        conductivity = st.number_input("Conductivity", min_value=0.0, value=400.0)

    col1, col2, col3 = st.columns(3)
    with col1:
        organic_carbon = st.number_input("Organic Carbon", min_value=0.0, value=10.0)
    with col2:
        trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, value=60.0)
    with col3:
        turbidity = st.number_input("Turbidity", min_value=0.0, value=4.0)

    # ===============================
    # 3. DataFrame Input
    # ===============================
    user_df = pd.DataFrame({
        "ph": [ph],
        "Hardness": [hardness],
        "Solids": [solids],
        "Chloramines": [chloramines],
        "Sulfate": [sulfate],
        "Conductivity": [conductivity],
        "Organic_carbon": [organic_carbon],
        "Trihalomethanes": [trihalomethanes],
        "Turbidity": [turbidity]
    })

    # ===============================
    # 4. Sinkronisasi Fitur
    # ===============================
    for col in feature_names:
        if col not in user_df.columns:
            user_df[col] = 0

    user_df = user_df[feature_names]

    # ===============================
    # 5. Normalisasi
    # ===============================
    scaler = MinMaxScaler()
    user_df[numeric_cols] = scaler.fit_transform(user_df[numeric_cols])

    # ===============================
    # 6. Prediction
    # ===============================
    if st.button("Prediksi Kelayakan"):
        prob = model.predict_proba(user_df)[0][1]
        pred = model.predict(user_df)[0]

        st.subheader("Hasil Prediksi")
        st.metric("Probabilitas Air Layak Minum", f"{prob * 100:.2f}%")

        if pred == 1:
            st.success("✅ Air **LAYAK DIKONSUMSI**")
        else:
            st.error("⚠️ Air **TIDAK LAYAK DIKONSUMSI**")

        st.divider()

        st.markdown("### Interpretasi")
        st.markdown("""
        - Probabilitas tinggi menunjukkan air cenderung **layak minum**
        - Model digunakan sebagai **alat bantu analisis awal**
        - Keputusan akhir tetap memerlukan **uji laboratorium resmi**
        """)


if __name__ == "__main__":
    prediction_app()
