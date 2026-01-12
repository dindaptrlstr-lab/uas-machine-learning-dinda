import streamlit as st
import pandas as pd
import altair as alt

def chart():
    # ===============================
    # Load Dataset
    # ===============================
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')

    pasien_count = df.shape[0]
    pasien_stroke = df['stroke'].sum()
    stroke_rate = (pasien_stroke / pasien_count) * 100

    # ===============================
    # Card Metrics & Filter Button
    # ===============================
    col1, col2, col3, col4, col5, col6 = st.columns([2,2,3,2,2,1])

    with col1:
        st.metric("Total Pasien", pasien_count)
    with col2:
        st.metric("Pasien Stroke", pasien_stroke)
    with col3:
        st.metric("Persentase Stroke", f"{stroke_rate:.2f}%")

    # Session State
    if 'selected_gender' not in st.session_state:
        st.session_state.selected_gender = None
    if 'selected_stroke' not in st.session_state:
        st.session_state.selected_stroke = None

    with col4:
        st.write("**Jenis Kelamin**")
        if st.button("Laki-laki"):
            st.session_state.selected_gender = 'Male'
        if st.button("Perempuan"):
            st.session_state.selected_gender = 'Female'

    with col5:
        st.write("**Status Stroke**")
        if st.button("Stroke"):
            st.session_state.selected_stroke = 1
        if st.button("No Stroke"):
            st.session_state.selected_stroke = 0

    with col6:
        if st.button("🔄"):
            st.session_state.selected_gender = None
            st.session_state.selected_stroke = None
            st.rerun()

    # ===============================
    # Apply Filter
    # ===============================
    filtered_df = df.copy()

    if st.session_state.selected_gender:
        filtered_df = filtered_df[filtered_df['gender'] == st.session_state.selected_gender]

    if st.session_state.selected_stroke is not None:
        filtered_df = filtered_df[filtered_df['stroke'] == st.session_state.selected_stroke]

    st.write("**Preview Data**")
    st.dataframe(filtered_df.head())

    # ===============================
    # Helper Pie Chart
    # ===============================
    def pie_chart(count_df, label, value, title):
        return alt.Chart(count_df).mark_arc(innerRadius=40).encode(
            theta=value,
            color=label,
            tooltip=[label, value]
        ).properties(title=title, height=300)

    # ===============================
    # Pie Charts
    # ===============================
    col1, col2 = st.columns(2)

    with col1:
        gender_counts = filtered_df['gender'].fillna('Unknown').value_counts().reset_index()
        gender_counts.columns = ['gender', 'count']
        st.altair_chart(
            pie_chart(gender_counts, 'gender:N', 'count:Q', 'Jenis Kelamin'),
            use_container_width=True
        )

    with col2:
        hyper_counts = filtered_df['hypertension'].map({0: 'Tidak', 1: 'Hipertensi'}).value_counts().reset_index()
        hyper_counts.columns = ['hypertension', 'count']
        st.altair_chart(
            pie_chart(hyper_counts, 'hypertension:N', 'count:Q', 'Hipertensi'),
            use_container_width=True
        )

    col1, col2 = st.columns(2)

    with col1:
        smoke_counts = filtered_df['smoking_status'].fillna('Unknown').value_counts().reset_index()
        smoke_counts.columns = ['smoking_status', 'count']
        st.altair_chart(
            pie_chart(smoke_counts, 'smoking_status:N', 'count:Q', 'Status Merokok'),
            use_container_width=True
        )

    with col2:
        residence_counts = filtered_df['Residence_type'].value_counts().reset_index()
        residence_counts.columns = ['Residence_type', 'count']
        st.altair_chart(
            pie_chart(residence_counts, 'Residence_type:N', 'count:Q', 'Tipe Tempat Tinggal'),
            use_container_width=True
        )

    # ===============================
    # Histogram BMI
    # ===============================
    st.write("**Distribusi BMI Pasien**")
    bmi_hist = alt.Chart(filtered_df).mark_bar().encode(
        x=alt.X('bmi:Q', bin=alt.Bin(maxbins=30), title='BMI'),
        y=alt.Y('count():Q', title='Jumlah Pasien'),
        tooltip=['count():Q']
    ).properties(height=300)
    st.altair_chart(bmi_hist, use_container_width=True)

    # ===============================
    # Scatter Plot
    # ===============================
    st.write("**BMI vs Rata-rata Glukosa**")
    scatter = alt.Chart(filtered_df).mark_circle(size=60).encode(
        x='avg_glucose_level:Q',
        y='bmi:Q',
        color='stroke:N',
        tooltip=['age', 'bmi', 'avg_glucose_level', 'stroke']
    ).interactive().properties(height=320)
    st.altair_chart(scatter, use_container_width=True)

    # ===============================
    # Boxplot Usia vs Stroke
    # ===============================
    st.write("**Distribusi Usia berdasarkan Status Stroke**")
    box = alt.Chart(filtered_df).mark_boxplot().encode(
        x='stroke:N',
        y='age:Q',
        color='stroke:N'
    ).properties(height=320)
    st.altair_chart(box, use_container_width=True)

    # ===============================
    # Line Chart Usia vs Stroke
    # ===============================
    st.write("**Distribusi Usia berdasarkan Kasus Stroke**")
    age_stroke = filtered_df.groupby(['age', 'stroke']).size().reset_index(name='count')
    age_stroke['stroke'] = age_stroke['stroke'].astype(str)

    line = alt.Chart(age_stroke).mark_line(point=True).encode(
        x='age:Q',
        y='count:Q',
        color='stroke:N',
        tooltip=['age', 'count', 'stroke']
    ).properties(height=350)

    st.altair_chart(line, use_container_width=True)
