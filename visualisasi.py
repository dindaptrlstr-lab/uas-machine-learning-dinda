import streamlit as st
import pandas as pd
import altair as alt

def chart():
    # ===============================
    # Load Dataset (SESUAI KAGGLE)
    # ===============================
    cardio = pd.read_csv("cardio_train.csv", sep=";")
    air = pd.read_csv("AirQualityUCI.csv", sep=";")

    # ===============================
    # Preprocessing Cardio
    # ===============================
    cardio['age_years'] = (cardio['age'] / 365).astype(int)
    cardio['gender'] = cardio['gender'].map({1: 'Perempuan', 2: 'Laki-laki'})

    total = cardio.shape[0]
    risk = cardio['cardio'].sum()
    risk_rate = (risk / total) * 100

    # ===============================
    # Metrics
    # ===============================
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Pasien", total)
    c2.metric("Pasien Berisiko", risk)
    c3.metric("Persentase Risiko", f"{risk_rate:.2f}%")

    st.write("### Preview Dataset Kardiovaskular")
    st.dataframe(cardio.head())

    # ===============================
    # Gender Pie
    # ===============================
    gender_count = cardio['gender'].value_counts().reset_index()
    gender_count.columns = ['gender', 'count']

    pie = alt.Chart(gender_count).mark_arc(innerRadius=50).encode(
        theta='count:Q',
        color='gender:N',
        tooltip=['gender', 'count']
    ).properties(title="Distribusi Gender")

    st.altair_chart(pie, use_container_width=True)

    # ===============================
    # Age vs Cardio
    # ===============================
    st.write("### Usia vs Risiko Kardiovaskular")

    box = alt.Chart(cardio).mark_boxplot().encode(
        x='cardio:N',
        y='age_years:Q',
        color='cardio:N'
    )

    st.altair_chart(box, use_container_width=True)

    # ===============================
    # Air Quality Preview
    # ===============================
    st.write("### Preview Dataset Air Quality")
    st.dataframe(air.head())
