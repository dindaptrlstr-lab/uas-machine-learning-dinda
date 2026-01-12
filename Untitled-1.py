import streamlit as st
import pandas as pd
import plotly.express as px

# ===============================
# Load Dataset
# ===============================
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

st.header("Exploratory Data Analysis (EDA)")
st.write("Dataset Stroke Prediction")

# ===============================
# 1. Dataset
# ===============================
st.write('**1. Preview Dataset**')
st.dataframe(df.head())

# ===============================
# 2. Distribusi Usia Pasien
# ===============================
st.write('**2. Distribusi Usia Pasien**')
age_dist = df['age'].value_counts().sort_index()
st.line_chart(age_dist)

# ===============================
# 3. Jenis Kelamin
# ===============================
st.write('**3. Distribusi Jenis Kelamin**')
gender_df = df['gender'].value_counts().reset_index()
gender_df.columns = ['Gender', 'Jumlah']
fig_gender = px.pie(
    gender_df,
    names='Gender',
    values='Jumlah',
    title='Distribusi Jenis Kelamin Pasien'
)
st.plotly_chart(fig_gender, use_container_width=True)

# ===============================
# 4. Status Stroke
# ===============================
st.write('**4. Status Stroke**')
stroke_df = df['stroke'].value_counts().reset_index()
stroke_df.columns = ['Stroke', 'Jumlah']
fig_stroke = px.bar(
    stroke_df,
    x='Stroke',
    y='Jumlah',
    title='Distribusi Pasien Stroke vs Non-Stroke'
)
st.plotly_chart(fig_stroke, use_container_width=True)

# ===============================
# 5. Tipe Pekerjaan
# ===============================
st.write('**5. Distribusi Tipe Pekerjaan**')
st.bar_chart(df['work_type'].value_counts())

# ===============================
# 6. Status Merokok
# ===============================
st.write('**6. Distribusi Status Merokok**')
st.bar_chart(df['smoking_status'].value_counts())

# ===============================
# 7. Penyakit Penyerta
# ===============================
st.write('**7. Penyakit Penyerta**')

col1, col2 = st.columns(2)

with col1:
    st.write('Hypertension')
    st.bar_chart(df['hypertension'].value_counts())

with col2:
    st.write('Heart Disease')
    st.bar_chart(df['heart_disease'].value_counts())

# ===============================
# 8. Filter Interaktif
# ===============================
st.write('**8. Filter Data Pasien**')

gender_filter = st.multiselect(
    "Pilih Jenis Kelamin",
    df['gender'].unique(),
    default=df['gender'].unique()
)

stroke_filter = st.selectbox(
    "Status Stroke",
    ['All', 0, 1]
)

filtered_df = df[df['gender'].isin(gender_filter)]

if stroke_filter != 'All':
    filtered_df = filtered_df[filtered_df['stroke'] == stroke_filter]

st.write("Data setelah difilter:")
st.dataframe(filtered_df)

# ===============================
# 9. Korelasi Numerik
# ===============================
st.write('**9. Korelasi Fitur Numerik**')
numeric_df = df[['age', 'avg_glucose_level', 'bmi', 'stroke']]
fig_corr = px.imshow(
    numeric_df.corr(),
    text_auto=True,
    title="Heatmap Korelasi"
)
st.plotly_chart(fig_corr, use_container_width=True)

# ===============================
# 10. Informasi Dataset
# ===============================
st.write('**10. Informasi Dataset**')
st.write(f"Jumlah Data: {df.shape[0]}")
st.write(f"Jumlah Fitur: {df.shape[1]}")

# ===============================
# 11. Gambar Ilustrasi
# ===============================
st.write('**11. Ilustrasi Kesehatan**')
st.image(
    "https://unair.ac.id/wp-content/uploads/2023/04/Foto-by-Kompas-Money.jpg",
    caption="Ilustrasi Rumah Sakit"
)
