import streamlit as st
import plotly.express as px
import pandas as pd


def dashboard_page():

    # =========================
    # PENGAMAN DATA (SESSION STATE)
    # =========================
    required_keys = ["df", "dataset_type", "target_col"]
    for key in required_keys:
        if key not in st.session_state:
            st.warning(
                "Silakan unggah dan pilih dataset terlebih dahulu "
                "pada menu Pemilihan Dataset."
            )
            return

    df = st.session_state["df"]
    dataset_type = st.session_state["dataset_type"]
    target_col = st.session_state["target_col"]

    # =========================
    # VALIDASI KOLOM TARGET
    # =========================
    if target_col not in df.columns:
        st.error("❌ Kolom target tidak ditemukan pada dataset.")
        st.write("Kolom yang tersedia:")
        st.write(list(df.columns))
        st.write("Kolom target yang dipilih:", target_col)
        return

    # =========================
    # JUDUL & DESKRIPSI HALAMAN
    # =========================
    st.subheader("Dashboard dan Eksplorasi Data (EDA)")

    st.markdown(f"""
    Halaman ini menyajikan hasil **Exploratory Data Analysis (EDA)**
    untuk dataset bertipe **{dataset_type}**.

    Tujuan utama EDA adalah untuk:
    - Mengetahui distribusi kelas pada variabel target
    - Mengidentifikasi potensi ketidakseimbangan data
    - Menganalisis hubungan antar fitur numerik

    Hasil analisis ini digunakan sebagai dasar
    sebelum dilakukan proses **pemodelan Machine Learning**.
    """)

    st.markdown("---")

    # =========================
    # RINGKASAN DATA
    # =========================
    total_data = len(df)
    positive_count = int(df[target_col].sum())
    positive_rate = (positive_count / total_data) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Data", total_data)
    col2.metric("Jumlah Target = 1", positive_count)
    col3.metric("Persentase Target = 1", f"{positive_rate:.2f}%")

    st.markdown("---")

    # =========================
    # DISTRIBUSI TARGET
    # =========================
    st.subheader("Distribusi Kelas Target")

    fig_target = px.pie(
        df,
        names=target_col,
        title=f"Distribusi Kelas Target ({target_col})"
    )
    st.plotly_chart(fig_target, use_container_width=True)

    st.markdown("""
    **Insight:**

    Visualisasi distribusi target menunjukkan proporsi
    masing-masing kelas pada dataset.

    Apabila distribusi kelas tidak seimbang,
    maka evaluasi model tidak cukup hanya menggunakan
    **akurasi**, tetapi perlu mempertimbangkan metrik lain
    seperti **presisi**, **recall**, dan **F1-score**.
    """)

    st.markdown("---")

    # =========================
    # KORELASI FITUR NUMERIK
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
        tingkat hubungan antar fitur numerik.

        - Nilai korelasi yang tinggi dapat mengindikasikan
          adanya **redundansi fitur**
        - Nilai korelasi yang rendah menunjukkan fitur
          yang relatif **independen**

        Informasi ini penting pada tahap
        seleksi fitur dan interpretasi model.
        """)
    else:
        st.info(
            "Dataset hanya memiliki satu fitur numerik, "
            "sehingga analisis korelasi tidak dapat dilakukan."
        )
