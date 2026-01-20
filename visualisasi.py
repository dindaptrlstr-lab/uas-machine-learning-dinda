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
        water = pd.read_csv("water_potability.csv")
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
    # Metrics Cardio
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
    st.subheader("Distribusi Gender Pasien")

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
    # WATER POTABILITY SECTION
    # ===============================
    st.header("Analisis Kualitas Air (Water Potability)")

    # ===============================
    # Preview Dataset Water
    # ===============================
    st.subheader("Preview Dataset Water Potability")
    st.dataframe(water.head(), use_container_width=True)

    # ===============================
    # Distribusi Potability
    # ===============================
    st.subheader("Distribusi Kelayakan Air Minum")

    pot_count = water["Potability"].value_counts().reset_index()
    pot_count.columns = ["Potability", "Jumlah"]

    pot_chart = alt.Chart(pot_count).mark_bar().encode(
        x=alt.X("Potability:N", title="Potability (0 = Tidak Layak, 1 = Layak)"),
        y=alt.Y("Jumlah:Q"),
        tooltip=["Potability", "Jumlah"],
        color="Potability:N"
    ).properties(title="Distribusi Air Layak Minum")

    st.altair_chart(pot_chart, use_container_width=True)

    # ===============================
    # Parameter Air vs Potability
    # ===============================
    st.subheader("Parameter Kualitas Air vs Potability")

    parameter = st.selectbox(
        "Pilih Parameter Air",
        [
            "ph", "Hardness", "Solids", "Chloramines",
            "Sulfate", "Conductivity", "Organic_carbon",
            "Trihalomethanes", "Turbidity"
        ]
    )

    box_water = alt.Chart(
        water.dropna()
    ).mark_boxplot().encode(
        x=alt.X("Potability:N", title="Potability"),
        y=alt.Y(f"{parameter}:Q", title=parameter),
        color="Potability:N"
    ).properties(
        title=f"{parameter} terhadap Kelayakan Air"
    )

    st.altair_chart(box_water, use_container_width=True)
