import streamlit as st
import pandas as pd
import os


def upload_page():

    # =========================
    # JUDUL HALAMAN
    # =========================
    st.subheader("Pemilihan Dataset")

    st.write(
        "Silakan pilih dataset yang akan digunakan sebagai dasar "
        "analisis dan pemodelan **Machine Learning**."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================
    # GAYA KARTU DATASET
    # =========================
    st.markdown("""
    <style>
    .dataset-card {
        background-color: #EEF4FB;
        padding: 24px;
        border-radius: 16px;
        height: 100%;
        border: 1px solid #E0E6ED;
    }

    .dataset-title {
        font-weight: 600;
        font-size: 17px;
        margin-bottom: 6px;
    }

    .dataset-desc {
        font-size: 14px;
        margin-bottom: 16px;
        color: #444444;
    }
    </style>
    """, unsafe_allow_html=True)

    # =========================
    # KONFIGURASI DATASET
    # =========================
    datasets = {
        "water": {
            "judul": "Dataset Kelayakan Air Minum",
            "deskripsi": "Dataset kualitas air untuk menentukan kelayakan air minum.",
            "target": "Potability",
            "tipe": "Lingkungan",
            "file": "water_potability.csv"
        },
        "cardio": {
            "judul": "Dataset Penyakit Kardiovaskular",
            "deskripsi": "Dataset data klinis untuk prediksi risiko penyakit jantung.",
            "target": "cardio",
            "tipe": "Kesehatan",
            "file": "cardio_train.csv"
        }
    }

    col1, col2 = st.columns(2)

    # =========================
    # KARTU DATASET LINGKUNGAN
    # =========================
    with col1:
        st.markdown(f"""
        <div class="dataset-card">
            <div class="dataset-title">{datasets['water']['judul']}</div>
            <div class="dataset-desc">{datasets['water']['deskripsi']}</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Gunakan Dataset Lingkungan", key="water_btn"):
            muat_dataset(datasets["water"])

    # =========================
    # KARTU DATASET KESEHATAN
    # =========================
    with col2:
        st.markdown(f"""
        <div class="dataset-card">
            <div class="dataset-title">{datasets['cardio']['judul']}</div>
            <div class="dataset-desc">{datasets['cardio']['deskripsi']}</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Gunakan Dataset Kesehatan", key="cardio_btn"):
            muat_dataset(datasets["cardio"])


def muat_dataset(konfigurasi):

    # =========================
    # PEMERIKSAAN FILE
    # =========================
    if not os.path.exists(konfigurasi["file"]):
        st.error(f"File `{konfigurasi['file']}` tidak ditemukan.")
        return

    # =========================
    # MEMUAT DATASET
    # =========================
    df = pd.read_csv(konfigurasi["file"], sep=None, engine="python")

    # =========================
    # RESET SESSION STATE LAMA
    # =========================
    for key in ["best_model", "scaler", "feature_columns"]:
        if key in st.session_state:
            del st.session_state[key]

    # =========================
    # SIMPAN KE SESSION STATE
    # =========================
    st.session_state["df"] = df
    st.session_state["dataset_name"] = konfigurasi["file"]
    st.session_state["target_col"] = konfigurasi["target"]
    st.session_state["dataset_type"] = konfigurasi["tipe"]

    # =========================
    # UMPAN BALIK KE PENGGUNA
    # =========================
    st.success(
        f"Dataset **{konfigurasi['judul']}** berhasil dimuat. "
        "Silakan lanjut ke menu berikutnya."
    )
