import streamlit as st
import plotly.express as px
import pandas as pd


def dashboard_page():

    # =========================
    # PENGAMAN SUPER KETAT
    # =========================
    required_keys = ["df", "dataset_type", "target_col"]
    for key in required_keys:
        if key not in st.session_state:
            st.warning("Silakan pilih data terlebih dahulu melalui menu pilih dataset.")
            return

    df = st.session_state["df"]
    dataset_type = st.session_state["dataset_type"]
    target_col = st.session_state["target_col"]

    # =========================
    # ANTI KEYERROR TARGET
    # =========================
    if target_col not in df.columns:
        st.error("❌ Target kolom tidak ditemukan pada dataset.")
        st.write("Kolom tersedia:")
        st.write(list(df.columns))
        st.write("Target yang dicari:", target_col)
        return

    # =========================
    # JUDUL & DESKRIPSI HALAMAN
    # =========================
    st.subheader("Exploratory Data Analysis (EDA) & Visualisasi")

    st.markdown(f"""
    Halaman ini menampilkan Exploratory Data Analysis (EDA) untuk dataset **{dataset_type}**.

    Tujuan EDA adalah untuk memahami:
    - Distribusi kelas target
    - Potensi ketidakseimbangan data
    - Hubungan antar fitur numerik

    Hasil analisis ini digunakan sebagai dasar
    sebelum dilakukan pemodelan Machine Learning.
    """)

    st.markdown("---")

    # =========================
    # METRIC RINGKASAN DATA
    # =========================
    total_data = len(df)
    positive_count = int(df[target_col].sum())
    positive_rate = (positive_count / total_data) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Data", total_data)
    col2.metric("Jumlah Target = 1", positive_count)
    col3.metric("Persentase Target = 1", f"{positive_rate:.2f}%")

    st.markdown("---")

    # =========================
    # DISTRIBUSI TARGET
    # =========================
    st.subheader("Distribusi Target")

    fig_target = px.pie(
        df,
        names=target_col,
        title=f"Distribusi Kelas Target `{target_col}`"
    )
    st.plotly_chart(fig_target, use_container_width=True)

    st.markdown("""
    **Insight:**

    Visualisasi distribusi target menunjukkan proporsi masing-masing kelas.
    Jika distribusi kelas tidak seimbang, maka evaluasi model
    tidak cukup hanya menggunakan **akurasi**, tetapi perlu
    mempertimbangkan **precision**, **recall**, dan **F1-score**.
    """)

    st.markdown("---")

    # =========================
    # HEATMAP KORELASI
    # =========================
    st.subheader("Korelasi Antar Fitur Numerik")

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] > 1:
        corr_matrix = numeric_df.corr()

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Heatmap Korelasi Fitur Numerik"
        )

        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("""
        **Insight:**

        Heatmap korelasi digunakan untuk mengidentifikasi
        hubungan antar fitur numerik.

        - Korelasi tinggi dapat mengindikasikan redundansi fitur
        - Korelasi rendah menunjukkan fitur yang lebih independen

        Informasi ini penting untuk tahap
        seleksi fitur dan interpretasi model.
        """)
    else:
        st.info(
            "Dataset hanya memiliki satu fitur numerik, "
            "sehingga analisis korelasi tidak dapat dilakukan."
        )


