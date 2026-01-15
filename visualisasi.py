import streamlit as st
import pandas as pd
import altair as alt


def chart():
    st.header("Dashboard Analisis Data")

    # ===============================
    # Load Dataset (SAFE)
    # ===============================
    try:
        cardio = pd.read_csv("cardio_train.csv", sep=";")
        air = pd.read_csv("AirQualityUCI.csv", sep=";")
    except Exception:
        st.error("❌ Dataset tidak ditemukan. Pastikan file CSV ada di folder proyek.")
        st.stop()

    # ===============================
    # Preprocessing Cardio
    # ===============================
    cardio["age_years"] = (cardio["age"] / 365).astype(int)
    cardio["gender"] = cardio["gender"].map({1: "Perempuan", 2: "Laki-laki"})

    total = cardio.shape[0]
    risk = int(cardio["cardio"].sum())
    risk_rate = (risk / total) * 100

    # ===============================
    # Metrics
    # ===============================
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Pasien", total)
    c2.metric("Pasien Berisiko", risk)
    c3.metric("Persentase Risiko", f"{risk_rate:.2f}%")

    st.divider()

    # ===============================
    # Preview Dataset Cardio
    # ===============================
    st.subheader("Preview Dataset Kardiovaskular")
    st.dataframe(cardio.head(), use_container_width=True)

    # ===============================
    # Gender Distribution (Pie)
    # ===============================
    st.subheader("Distribusi Gender")

    gender_count = cardio["gender"].value_counts().reset_index()
    gender_count.columns = ["Gender", "Jumlah"]

    pie = alt.Chart(gender_count).mark_arc(innerRadius=50).encode(
        theta="Jumlah:Q",
        color="Gender:N",
        tooltip=["Gender", "Jumlah"]
    ).properties(title="Distribusi Gender Pasien")

    st.altair_chart(pie, use_container_width=True)

    # ===============================
    # Age vs Cardio (Boxplot)
    # ===============================
    st.subheader("Usia vs Risiko Kardiovaskular")

    box = alt.Chart(cardio).mark_boxplot().encode(
        x=alt.X("cardio:N", title="Status Kardiovaskular (0 = Tidak, 1 = Ya)"),
        y=alt.Y("age_years:Q", title="Usia (Tahun)"),
        color="cardio:N"
    )

    st.altair_chart(box, use_container_width=True)

    st.divider()

    # ===============================
    # Air Quality Preview
    # ===============================
    st.subheader("Preview Dataset Air Quality")
    st.dataframe(air.head(), use_container_width=True)
