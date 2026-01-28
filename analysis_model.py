import streamlit as st 
import pandas as pd
import numpy as np
from scipy.special import expit  # sigmoid stabil numerik


def analysis_model_page():

    # =====================================================
    # PENGAMAN DATASET
    # =====================================================
    if "df" not in st.session_state or "dataset_name" not in st.session_state:
        st.warning("Silakan pilih dataset terlebih dahulu pada menu Upload Data.")
        return

    df = st.session_state["df"]
    dataset_name = st.session_state["dataset_name"]

    # =====================================================
    # TARGET OTOMATIS BERDASARKAN DATASET
    # =====================================================
    if dataset_name == "water_potability.csv":
        target_col = "Potability"
        dataset_type = "Lingkungan"
    elif dataset_name == "cardio_train.csv":
        target_col = "cardio"
        dataset_type = "Kesehatan"
    else:
        st.error("Dataset tidak dikenali. Silakan gunakan dataset yang tersedia.")
        return

    # =====================================================
    # JUDUL & DESKRIPSI HALAMAN
    # =====================================================
    st.subheader("Analisis Model Klasifikasi (Detail Perhitungan)")

    st.write(
        f"Dataset: **{dataset_name}** ({dataset_type})  \n"
        "Halaman ini menampilkan **mekanisme internal dan perhitungan inti** "
        "dari setiap algoritma klasifikasi."
    )

    st.markdown("""
    Halaman ini dirancang untuk **tujuan edukatif**,
    yaitu menjelaskan **cara kerja algoritma Machine Learning**
    secara **konseptual dan matematis**.
    """)

    st.markdown("---")

    # =====================================================
    # PILIH ALGORITMA
    # =====================================================
    algo = st.selectbox(
        "Pilih Algoritma Klasifikasi",
        [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "Support Vector Machine (SVM)",
            "CatBoost"
        ]
    )

    st.markdown("---")

    # =====================================================
    # SAMPLE DATA NUMERIK
    # =====================================================
    numeric_df = df.select_dtypes(include="number").drop(
        columns=[target_col], errors="ignore"
    )

    if numeric_df.empty:
        st.error("Dataset tidak memiliki fitur numerik yang dapat dianalisis.")
        return

    X_sample = numeric_df.iloc[0].values

    # =====================================================
    # LOGISTIC REGRESSION
    # =====================================================
    if algo == "Logistic Regression":
        st.subheader("Logistic Regression")

        st.markdown("""
        **Tahapan Logistic Regression:**
        1. Menghitung kombinasi linear fitur  
           \\[
           z = \\beta_0 + \\beta_1x_1 + \\beta_2x_2 + ... + \\beta_nx_n
           \\]
        2. Mengubah nilai `z` menjadi probabilitas menggunakan fungsi sigmoid  
           \\[
           P(y=1) = \\frac{1}{1 + e^{-z}}
           \\]
        3. Menentukan kelas berdasarkan threshold (0.5)
        4. Menghitung kesalahan prediksi menggunakan Log Loss  
           \\[
           L = -[y \\log(p) + (1-y) \\log(1-p)]
           \\]
        """)

        beta = np.ones(len(X_sample)) * 0.1
        beta_0 = 0.1

        z = beta_0 + np.dot(X_sample, beta)
        prob = expit(z)
        kelas = 1 if prob >= 0.5 else 0
        log_loss = -(np.log(prob) if kelas == 1 else np.log(1 - prob))

        st.write("**Model:**  z = β₀ + β·x")
        st.write(f"Nilai z = `{z:.4f}`")
        st.write(f"Sigmoid(z) = `{prob:.4f}`")
        st.write(f"Prediksi Kelas = `{kelas}`")
        st.write(f"Log Loss (1 data) = `{log_loss:.4f}`")

    # =====================================================
    # DECISION TREE
    # =====================================================
    elif algo == "Decision Tree":
        st.subheader("Decision Tree")

        st.markdown("""
        **Tahapan Decision Tree:**
        1. Menghitung Entropy awal dataset  
           \\[
           Entropy = -\\sum p_i \\log_2(p_i)
           \\]
        2. Memilih fitur dan threshold sebagai pemisah
        3. Menghitung Information Gain  
           \\[
           IG = Entropy_{awal} - Entropy_{setelah\\ split}
           \\]
        4. Membentuk cabang hingga kondisi berhenti
        """)

        class_prob = df[target_col].value_counts(normalize=True)
        entropy = -(class_prob * np.log2(class_prob)).sum()

        feature = numeric_df.columns[0]
        threshold = df[feature].median()

        left = df[df[feature] <= threshold]
        right = df[df[feature] > threshold]

        def entropy_subset(data):
            p = data[target_col].value_counts(normalize=True)
            return -(p * np.log2(p)).sum() if len(p) > 0 else 0

        information_gain = entropy - (
            (len(left) / len(df)) * entropy_subset(left) +
            (len(right) / len(df)) * entropy_subset(right)
        )

        st.write(f"Entropy Awal = `{entropy:.4f}`")
        st.write(f"Fitur Contoh = `{feature}`")
        st.write(f"Information Gain = `{information_gain:.4f}`")

    # =====================================================
    # RANDOM FOREST
    # =====================================================
    elif algo == "Random Forest":
        st.subheader("Random Forest")

        st.markdown("""
        **Tahapan Random Forest:**
        1. Melakukan bootstrap sampling dari dataset
        2. Membangun banyak Decision Tree
        3. Setiap tree dilatih dengan subset fitur acak
        4. Prediksi akhir ditentukan dengan majority voting
        """)

        fake_tree_predictions = np.random.choice(
            df[target_col].unique(), size=7
        )

        vote_result = pd.Series(fake_tree_predictions).value_counts()

        st.write("Prediksi dari tiap tree:")
        st.write(fake_tree_predictions)
        st.success(f"Hasil Voting Mayoritas = `{vote_result.idxmax()}`")

    # =====================================================
    # SUPPORT VECTOR MACHINE
    # =====================================================
    elif algo == "Support Vector Machine (SVM)":
        st.subheader("Support Vector Machine (SVM)")

        st.markdown("""
        **Tahapan SVM:**
        1. Menentukan hyperplane pemisah  
           \\[
           f(x) = w \\cdot x + b
           \\]
        2. Memaksimalkan margin antar kelas
        3. Kelas ditentukan berdasarkan tanda fungsi keputusan
        """)

        w = np.ones(len(X_sample)) * 0.5
        b = -0.2

        decision_value = np.dot(w, X_sample) + b
        kelas = 1 if decision_value >= 0 else 0

        st.write(f"f(x) = `{decision_value:.4f}`")
        st.success(f"Prediksi Kelas = `{kelas}`")

    # =====================================================
    # CATBOOST
    # =====================================================
    elif algo == "CatBoost":
        st.subheader("CatBoost")

        st.markdown("""
        **Tahapan CatBoost:**
        1. Membuat prediksi awal
        2. Menghitung error (residual)
        3. Membangun tree baru untuk memperbaiki error
        4. Memperbarui prediksi  
           \\[
           F_{baru} = F_{lama} + \\eta \\times error
           \\]
        """)

        initial_prediction = 0.5
        learning_rate = 0.1
        error_correction = -0.2

        updated_prediction = initial_prediction + learning_rate * error_correction

        st.write(f"Prediksi Awal = `{initial_prediction}`")
        st.write(f"Prediksi Baru = `{updated_prediction:.4f}`")

    # =====================================================
    # CATATAN PENUTUP
    # =====================================================
    st.markdown("---")
    st.info(
        "Penjelasan tahapan dan rumus di atas digunakan untuk **memahami konsep algoritma**. "
        "Training dan evaluasi model secara nyata dilakukan pada menu **Machine Learning**."
    )
