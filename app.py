import streamlit as st

# ======================
# IMPORT HALAMAN
# ======================
from upload import upload_page
from about import show_about
from dashboard import dashboard_page
from modeling import modeling_page
from analysis_model import analysis_model_page
from prediction import prediction_page
from contact import contact_page


# ======================
# KONFIGURASI HALAMAN
# ======================
st.set_page_config(
    page_title="Dashboard Klasifikasi Machine Learning 
    untuk Analisis Data Kesehatan dan Lingkungan",
    layout="wide"
)

# ======================
# HILANGKAN SIDEBAR TOTAL (CSS)
# ======================
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)


# ======================
# HEADER UTAMA
# ======================
st.title("Dashboard Klasifikasi Machine Learning untuk Analisis Data Kesehatan dan Lingkungan")

st.caption(
    "Dashboard untuk analisis dan klasifikasi data kesehatan serta lingkungan "
    "dengan pendekatan Machine Learning."
)

st.markdown("---")

st.info(
    "Proyek Akhir UAS – Mata Kuliah Machine Learning | "
    "Program Studi Sains Data"
)

# ======================
# TAB NAVIGASI (ALUR ML)
# ======================
tabs = st.tabs([
    "Pilih Dataset",
    "About Dataset",
    "Exploratory Data Analysis",
    "Machine Learning",
    "Tahapan Model",
    "Prediction",
    "Contact"
])

# ======================
# ISI TAB
# ======================
with tabs[0]:
    upload_page()

with tabs[1]:
    show_about()

with tabs[2]:
    dashboard_page()

with tabs[3]:
    modeling_page()

with tabs[4]:
    analysis_model_page()

with tabs[5]:
    prediction_page()

with tabs[6]:
    contact_page()





