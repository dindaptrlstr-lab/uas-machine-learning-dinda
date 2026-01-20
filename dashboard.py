import streamlit as st
import plotly.express as px
import pandas as pd

def dashboard_page():

    # ===== PENGAMAN =====
    if "df" not in st.session_state or "dataset_type" not in st.session_state:
        st.warning("Silakan upload dataset terlebih dahulu.")
        return

    df = st.session_state["df"]
    dataset_type = st.session_state["dataset_type"]
    target_col = st.session_state["target_col"]

    st.title("📊 Dashboards & Exploratory Data Analysis")

    # =========================================================
    # 🔍 EDA UMUM
    # =========================================================
    st.subheader("🔍 Exploratory Data Analysis (Umum)")

    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Data", df.shape[0])
    col2.metric("Jumlah Fitur", df.shape[1])
    col3.metric("Missing Value", df.isnull().sum().sum())

    st.write("### 📌 Tipe Data")
    st.dataframe(df.dtypes.astype(str), use_container_width=True)

    st.write("### 📊 Statistik Deskriptif")
    st.dataframe(df.describe(), use_container_width=True)

    # ===== HEATMAP KORELASI =====
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] > 1:
        corr = numeric_df.corr()
        fig = px.imshow(
            corr,
            text_auto=True,
            title="Heatmap Korelasi Fitur Numerik"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # =========================================================
    # 🌱 DATASET LINGKUNGAN – WATER POTABILITY
    # =========================================================
    if dataset_type == "Lingkungan":

        st.subheader("🌱 Dashboard Lingkungan – Water Potability")

        total_data = df.shape[0]
        potable = df[target_col].sum()
        rate = (potable / total_data) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Sampel Air", total_data)
        col2.metric("Air Layak Minum", potable)
        col3.metric("Persentase Layak", f"{rate:.2f}%")

        st.dataframe(df.head(), use_container_width=True)

        # ===== PIE TARGET =====
        fig = px.pie(
            df,
            names=target_col,
            title="Distribusi Kelayakan Air (0 = Tidak Layak, 1 = Layak)"
        )
        st.plotly_chart(fig, use_container_width=True)

        # ===== HISTOGRAM =====
        fig = px.histogram(
            df,
            x="ph",
            nbins=30,
            title="Distribusi pH Air"
        )
        st.plotly_chart(fig, use_container_width=True)

        # ===== SCATTER =====
        fig = px.scatter(
            df,
            x="Solids",
            y="Turbidity",
            color=target_col,
            title="Solids vs Turbidity"
        )
        st.plotly_chart(fig, use_container_width=True)

        # ===== BOX PLOT =====
        fig = px.box(
            df,
            x=target_col,
            y="Hardness",
            title="Hardness berdasarkan Kelayakan Air"
        )
        st.plotly_chart(fig, use_container_width=True)

    # =========================================================
    # 🏥 DATASET KESEHATAN – CARDIOVASCULAR
    # =========================================================
    elif dataset_type == "Kesehatan":

        st.subheader("🏥 Dashboard Kesehatan – Cardiovascular Disease")

        total_data = df.shape[0]
        risk = df[target_col].sum()
        rate = (risk / total_data) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Pasien", total_data)
        col2.metric("Pasien Berisiko", risk)
        col3.metric("Persentase Risiko", f"{rate:.2f}%")

        st.dataframe(df.head(), use_container_width=True)

        # ===== PIE TARGET =====
        fig = px.pie(
            df,
            names=target_col,
            title="Distribusi Risiko Penyakit Jantung"
        )
        st.plotly_chart(fig, use_container_width=True)

        # ===== HISTOGRAM =====
        fig = px.histogram(
            df,
            x="age",
            nbins=30,
            title="Distribusi Usia Pasien"
        )
        st.plotly_chart(fig, use_container_width=True)

        # ===== SCATTER =====
        fig = px.scatter(
            df,
            x="ap_hi",
            y="ap_lo",
            color=target_col,
            title="Tekanan Darah Sistolik vs Diastolik"
        )
        st.plotly_chart(fig, use_container_width=True)

        # ===== BOX PLOT =====
        fig = px.box(
            df,
            x=target_col,
            y="cholesterol",
            title="Kolesterol berdasarkan Risiko"
        )
        st.plotly_chart(fig, use_container_width=True)
