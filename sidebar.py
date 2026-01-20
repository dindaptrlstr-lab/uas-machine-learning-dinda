import streamlit as st
import pandas as pd

def sidebar_upload():
    st.sidebar.title("📂 Upload Dataset")

    uploaded_file = st.sidebar.file_uploader(
        "Upload file CSV",
        type=["csv"]
    )

    if uploaded_file:
        if uploaded_file.name not in [
            "water_potability.csv",
            "cardio_train.csv"
        ]:
            st.sidebar.error(
                "Dataset tidak valid.\n"
                "Gunakan:\n"
                "- water_potability.csv\n"
                "- cardio_train.csv"
            )
            return

        df = pd.read_csv(uploaded_file)

        # SIMPAN GLOBAL
        st.session_state["df"] = df
        st.session_state["dataset_name"] = uploaded_file.name

        st.sidebar.success("Dataset berhasil dimuat ✅")

        # SET TARGET OTOMATIS
        if uploaded_file.name == "water_potability.csv":
            st.session_state["target_col"] = "Potability"
            st.session_state["dataset_type"] = "Lingkungan"

        elif uploaded_file.name == "cardio_train.csv":
            st.session_state["target_col"] = "cardio"
            st.session_state["dataset_type"] = "Kesehatan"
