import streamlit as st
import pandas as pd
import numpy as np

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Cardiovascular & Air Quality Analysis",
    layout="wide"
)

# ===============================
# Header
# ===============================
st.title("❤️ Cardiovascular Disease & 🌫️ Air Quality Analysis")
st.caption(
    "Aplikasi Analisis Data Kesehatan dan Lingkungan "
    "Menggunakan Machine Learning"
)
st.caption("📍 Semarang, 12 Januari 2026")

st.divider()

# ===============================
# Tabs
# ===============================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 About Dataset",
    "📊 Dashboards",
    "🤖 Machine Learning",
    "🔮 Prediction App",
    "📬 Contact Me"
])

# ===============================
# About Dataset
# ===============================
with tab1:
    import about
    about.about_dataset()

# ===============================
# Dashboard / EDA
# ===============================
with tab2:
    import visualisasi
    visualisasi.chart()

# ===============================
# Machine Learning
# ===============================
with tab3:
    import machine_learning
    machine_learning.ml_model()

# ===============================
# Prediction App
# ===============================
with tab4:
    import prediction
    prediction.prediction_app()

# ===============================
# Contact Me
# ===============================
with tab5:
    st.subheader("📬 Contact Me")
    st.write("📧 Email : **dindaptrlstr@email.com**")
    st.write("💻 GitHub : https://github.com/dindaptrlstr-lab")
