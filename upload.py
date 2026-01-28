import streamlit as st
import pandas as pd
import os


def upload_page():

    # =========================
    # JUDUL HALAMAN
    # =========================
    st.subheader("Pilih Dataset")

    st.write(
        "Pilih dataset yang akan digunakan sebagai dasar "
        "analisis dan pemodelan **Machine Learning**."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================
    # STYLE CARD (NETRAL, NYATU DENGAN BANNER)
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
        margin-bottom: 14px;
        color: #444444;
    }
    </style>
    """, unsafe_allow_html=True)

    # =========================
    # KONFIGURASI DATASET (LOGIKA)
    # =========================
    datasets = {
        "water": {
            "title": "Water Potability Dataset",
            "desc": "Dataset kualitas air untuk menentukan kelayakan air minum.",
            "target": "Potability",
            "type": "Lingkungan",
            "file": "water_potability.csv"
        },
        "cardio": {
            "title": "Cardiovascular Disease Dataset",
            "desc": "Dataset klinis untuk prediksi risiko penyakit jantung.",
            "target": "cardio",
            "type": "Kesehatan",
            "file": "cardio_train.csv"
        }
    }

    col1, col2 = st.columns(2)

    # =========================
    # CARD DATASET AIR
    # =========================
    with col1:
        st.markdown(f"""
        <div class="dataset-card">
            <div class="dataset-title">{datasets['water']['title']}</div>
            <div class="dataset-desc">{datasets['water']['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Gunakan Dataset Lingkungan", key="water_btn"):
            load_dataset(datasets["water"])

    # =========================
    # CARD DATASET KESEHATAN
    # =========================
    with col2:
        st.markdown(f"""
        <div class="dataset-card">
            <div class="dataset-title">{datasets['cardio']['title']}</div>
            <div class="dataset-desc">{datasets['cardio']['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Gunakan Dataset Kesehatan", key="cardio_btn"):
            load_dataset(datasets["cardio"])


def load_dataset(config):

    # =========================
    # CEK FILE
    # =========================
    if not os.path.exists(config["file"]):
        st.error(f"File `{config['file']}` tidak ditemukan.")
        return

    # =========================
    # LOAD DATA
    # =========================
    df = pd.read_csv(config["file"], sep=None, engine="python")

    # =========================
    # RESET STATE LAMA
    # =========================
    for key in ["best_model", "scaler", "feature_columns"]:
        if key in st.session_state:
            del st.session_state[key]

    # =========================
    # SIMPAN KE SESSION STATE
    # (LOGIKA ML TETAP UTUH)
    # =========================
    st.session_state["df"] = df
    st.session_state["dataset_name"] = config["file"]
    st.session_state["target_col"] = config["target"]
    st.session_state["dataset_type"] = config["type"]

    # =========================
    # FEEDBACK
    # =========================
    st.success(f"Dataset **{config['title']}** berhasil dimuat")

    with st.expander("Lihat 5 baris pertama dataset"):
        st.dataframe(df.head(), use_container_width=True)
