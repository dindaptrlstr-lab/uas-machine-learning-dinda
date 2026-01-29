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
    page_title="Machine Learning Classification Dashboard",
    layout="wide"
)


# ======================
# HILANGKAN SIDEBAR TOTAL
# ======================
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)


# ======================
# HEADER UTAMA (CENTER)
# ======================
st.markdown("""
<div style="text-align:center; padding-top:15px;">
    <h1>Machine Learning Classification Dashboard</h1>
    <p style="font-size:16px;">
        Dashboard untuk analisis dan klasifikasi data kesehatan serta lingkungan dengan pendekatan Machine Learning
    </p>
    <hr style="width:60%; margin:auto;">
    <div style="
        margin-top:12px;
        padding:10px 20px;
        background-color:#eef6ff;
        border-radius:8px;
        display:inline-block;
        font-size:16px;
    ">
        <b>Proyek Akhir UAS</b> – Mata Kuliah Machine Learning | Program Studi Sains Data
    </div>
</div>
""", unsafe_allow_html=True)


# ======================
# JUDUL NAVIGASI (CENTER)
# ======================
st.markdown("""
<h3 style="text-align:center; margin-top:30px;">
    Navigasi Dashboard
</h3>
""", unsafe_allow_html=True)


# ======================
# TAB NAVIGASI
# ======================
tabs = st.tabs([
    "Pilih Dataset",
    "Tentang Dataset",
    "Eksplorasi Data",
    "Machine Learning",
    "Tahapan Pemodelan",
    "Prediksi",
    "Kontak"
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

