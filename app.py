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
# HILANGKAN SIDEBAR + STYLE TAB
# ======================
st.markdown("""
<style>
/* Hilangkan sidebar */
section[data-testid="stSidebar"] {
    display: none;
}

/* Tabs full width */
div[data-testid="stTabs"] {
    width: 100%;
}

div[data-testid="stTabs"] div[role="tablist"] {
    display: flex;
    justify-content: space-between;
    width: 100%;
}

div[data-testid="stTabs"] button[role="tab"] {
    flex-grow: 1;
    text-align: center;
    white-space: nowrap;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)


# ======================
# HEADER UTAMA (CENTER + BANNER BIRU)
# ======================
st.markdown("""
<div style="text-align:center; padding-top:20px;">

    <h1>Machine Learning Classification Dashboard</h1>

    <div style="
        background-color:#e3f2fd;
        color:#0d47a1;
        padding:14px 0;
        margin:12px 0;
        font-size:16px;
        width:100%;
    ">
        Dashboard untuk analisis dan klasifikasi data kesehatan serta lingkungan
        dengan pendekatan <b>Machine Learning</b>
    </div>

    <hr style="width:60%; margin:20px auto;">
</div>
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
