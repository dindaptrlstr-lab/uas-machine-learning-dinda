import streamlit as st
import pandas as pd
import plotly.express as px

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Health & Air Quality Analysis",
    layout="wide"
)

st.title("Cardiovascular Disease & Air Quality Analysis")

# ===============================
# Load Dataset
# ===============================
@st.cache_data
def load_cardio():
    return pd.read_csv("cardiovascular.csv", sep=";")

@st.cache_data
def load_air():
    return pd.read_csv("air_quality.csv", sep=";", decimal=",")

try:
    cardio_df = load_cardio()
    air_df = load_air()
except:
    st.error("❌ Dataset tidak ditemukan. Pastikan file CSV ada di repo.")
    st.stop()

# ===============================
# SECTION 1: CARDIOVASCULAR
# ===============================
st.header("📊 Cardiovascular Disease Dataset")

st.subheader("1️⃣ Preview Dataset")
st.dataframe(cardio_df.head(), use_container_width=True)

# ===============================
# Distribusi Umur
# ===============================
st.subheader("2️⃣ Distribusi Usia Pasien")

cardio_df["age_years"] = cardio_df["age"] // 365

st.line_chart(
    cardio_df["age_years"].value_counts().sort_index()
)

# ===============================
# Status Penyakit Jantung
# ===============================
st.subheader("3️⃣ Status Penyakit Kardiovaskular")

cardio_target = cardio_df["cardio"].value_counts().reset_index()
cardio_target.columns = ["Cardio", "Jumlah"]

fig_cardio = px.bar(
    cardio_target,
    x="Cardio",
    y="Jumlah",
    text="Jumlah"
)
st.plotly_chart(fig_cardio, use_container_width=True)

# ===============================
# Tekanan Darah
# ===============================
st.subheader("4️⃣ Tekanan Darah")

fig_bp = px.scatter(
    cardio_df,
    x="ap_hi",
    y="ap_lo",
    color="cardio",
    labels={"ap_hi": "Sistolik", "ap_lo": "Diastolik"}
)
st.plotly_chart(fig_bp, use_container_width=True)

# ===============================
# SECTION 2: AIR QUALITY
# ===============================
st.header("Air Quality Dataset")

st.subheader("5️⃣ Preview Dataset")
st.dataframe(air_df.head(), use_container_width=True)

# ===============================
# Cleaning Air Quality
# ===============================
air_df.replace(-200, pd.NA, inplace=True)
air_df.dropna(inplace=True)

# ===============================
# Polutan Udara
# ===============================
st.subheader("6️⃣ Distribusi Polutan Udara")

pollutant = st.selectbox(
    "Pilih Polutan",
    ["CO(GT)", "NO2(GT)", "NOx(GT)", "C6H6(GT)"]
)

fig_pollutant = px.line(
    air_df,
    y=pollutant,
    title=f"Tren {pollutant}"
)
st.plotly_chart(fig_pollutant, use_container_width=True)

# ===============================
# Korelasi Polutan
# ===============================
st.subheader("7️⃣ Korelasi Polutan")

pollution_cols = [
    "CO(GT)", "NO2(GT)", "NOx(GT)", "C6H6(GT)", "T", "RH"
]

corr_df = air_df[pollution_cols]

fig_corr = px.imshow(
    corr_df.corr(),
    text_auto=".2f",
    aspect="auto"
)
st.plotly_chart(fig_corr, use_container_width=True)

# ===============================
# INSIGHT
# ===============================
st.header("📌 Insight Awal")

st.markdown("""
- Dataset **Cardiovascular** cocok untuk **Logistic Regression, Random Forest, SVM**
- Dataset **Air Quality** cocok untuk **Regresi & Time Series**
- Bisa dikombinasikan untuk analisis **pengaruh kualitas udara terhadap risiko kardiovaskular**
""")
