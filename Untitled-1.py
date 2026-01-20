import streamlit as st
import pandas as pd
import plotly.express as px

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Health & Water Quality Analysis",
    layout="wide"
)

st.title("Cardiovascular Disease & Water Quality Analysis")

# ===============================
# Load Dataset
# ===============================
@st.cache_data
def load_cardio():
    return pd.read_csv("cardiovascular.csv", sep=";")

@st.cache_data
def load_water():
    return pd.read_csv("water_potability.csv")

try:
    cardio_df = load_cardio()
    water_df = load_water()
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
# SECTION 2: WATER QUALITY
# ===============================
st.header("💧 Water Potability Dataset")

st.subheader("5️⃣ Preview Dataset")
st.dataframe(water_df.head(), use_container_width=True)

# ===============================
# Cleaning Water Quality
# ===============================
st.subheader("6️⃣ Missing Value Check")
st.write(water_df.isna().sum())

water_df_clean = water_df.dropna()

# ===============================
# Distribusi Target Potability
# ===============================
st.subheader("7️⃣ Distribusi Air Layak Minum")

potability_count = water_df_clean["Potability"].value_counts().reset_index()
potability_count.columns = ["Potability", "Jumlah"]

fig_pot = px.bar(
    potability_count,
    x="Potability",
    y="Jumlah",
    text="Jumlah",
    labels={"Potability": "Air Layak Minum (1 = Ya, 0 = Tidak)"}
)
st.plotly_chart(fig_pot, use_container_width=True)

# ===============================
# Parameter Air
# ===============================
st.subheader("8️⃣ Distribusi Parameter Kualitas Air")

parameter = st.selectbox(
    "Pilih Parameter Air",
    [
        "ph", "Hardness", "Solids", "Chloramines",
        "Sulfate", "Conductivity", "Organic_carbon",
        "Trihalomethanes", "Turbidity"
    ]
)

fig_param = px.box(
    water_df_clean,
    y=parameter,
    color="Potability",
    title=f"Distribusi {parameter} terhadap Potability"
)
st.plotly_chart(fig_param, use_container_width=True)

# ===============================
# Korelasi
# ===============================
st.subheader("9️⃣ Korelasi Parameter Air")

corr = water_df_clean.corr()

fig_corr = px.imshow(
    corr,
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
- Dataset **Water Potability** cocok untuk **Classification & Feature Importance**
- Bisa dikombinasikan sebagai studi **lingkungan & kesehatan masyarakat**
""")
