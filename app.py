import streamlit as st
import pandas as pd
import numpy as np

st.header('Cardiovascular Disease & Air Quality Analysis')
st.write('**Aplikasi Analisis Data Kesehatan dan Lingkungan Menggunakan Machine Learning**')
st.write('Semarang, 12 Januari 2026')

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    'About Dataset', 
    'Dashboards', 
    'Machine Learning',
    'Prediction App',
    'Contact Me'
])

with tab1:
    import about
    about.about_dataset()

with tab2:
    import visualisasi
    visualisasi.chart()

with tab3:
    import machine_learning
    machine_learning.ml_model()

with tab4:
    import prediction
    prediction.prediction_app()

with tab5:
    st.write("📧 Email  : dindaptrlstr@email.com")
    st.write("📍 Kampus : Universitas Muhammadiyah Semarang")
