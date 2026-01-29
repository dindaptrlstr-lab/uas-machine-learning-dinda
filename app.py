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

/* ===== TAB FULL WIDTH ===== */
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
# HEADER UTAMA (CENTER)
# ======================
st.markdown("""
<div style="text-align:center; padding-top:15px;">
    <h1>Machine Learning Classification Dashboard</h1>

    <div style="
        margin:15px auto;
        width:70%;
        background-color:#F2F6FB;
        padding:18px;
        border-radius:16px;
    ">
        <p style="font-size:16px; margin:0;">
            Dashboard untuk analisis dan klasifikasi data kesehatan serta lingkungan
            dengan pendekatan <b>Machine Learning</b>
        </p>
    </div>

    <hr style="width:60%; margin:auto;">
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


