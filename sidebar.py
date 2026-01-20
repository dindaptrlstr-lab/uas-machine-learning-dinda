import streamlit as st
import pandas as pd

def sidebar_upload():
    st.sidebar.title("📂 Upload Dataset")

    uploaded_file = st.sidebar.file_uploader(
        "Upload file CSV",
        type=["csv"]
    )

    if uploaded_file:
        # 🔥 AUTO-DETECT DELIMITER
        df = pd.read_csv(uploaded_file, sep=None, engine="python")

        st.session_state["df"] = df
        st.session_state["dataset_name"] = uploaded_file.name

        if uploaded_file.name == "water_potability.csv":
            st.session_state["target_col"] = "Potability"
            st.session_state["dataset_type"] = "Lingkungan"

        elif uploaded_file.name == "cardio_train.csv":
            st.session_state["target_col"] = "cardio"
            st.session_state["dataset_type"] = "Kesehatan"

        else:
            st.sidebar.error("Dataset tidak dikenali")
            return

        st.sidebar.success("Dataset berhasil dimuat ✅")
