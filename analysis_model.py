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
    # TARGET OTOMATIS
    # =====================================================
    if dataset_name == "water_potability.csv":
        target_col = "Potability"
        dataset_type = "Lingkungan"
    elif dataset_name == "cardio_train.csv":
        target_col = "cardio"
        dataset_type = "Kesehatan"
    else:
        st.error("Dataset tidak dikenali.")
        return

    # =====================================================
    # JUDUL HALAMAN
    # =====================================================
    st.subheader("Analisis Model Klasifikasi (Tahapan & Perhitungan)")

    st.write(
        f"Dataset: **{dataset_name}** ({dataset_type})  \n"
        "Halaman ini menjelaskan **tahapan kerja dan perhitungan inti** "
        "dari setiap algoritma Machine Learning klasifikasi."
    )

    st.markdown("""
    Penjelasan pada halaman ini bersifat **edukatif** dan bertujuan
    untuk membantu memahami **alur logika algoritma**.
    Perhitungan yang ditampilkan merupakan **simulasi sederhana**,
    bukan hasil training aktual model.
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
        st.error("Dataset tidak memiliki fitur numerik.")
        return

    X_sample = numeric_df.iloc[0].values

    # =====================================================
    # LOGISTIC REGRESSION
    # =====================================================
    if algo == "Logistic Regression":
        st.subheader("Logistic Regression")

        st.markdown("""
        **Tahapan Logistic Regression:**
        1. Menghitung kombinasi linear antara fitur dan parameter (β)
        2. Mengubah nilai linear menjadi probabilitas menggunakan fungsi sigmoid
        3. Menentukan kelas berdasarkan ambang batas (threshold)
        4. Mengukur error menggunakan fungsi log loss
        """)

        beta = np.ones(len(X_sample)) * 0.1
        beta_0 = 0.1

        z = beta_0 + np.dot(X_sample, beta)
        prob = expit(z)
        kelas = 1 if prob >= 0.5 else 0
        log_loss = -(np.log(prob) if kelas == 1 else np.log(1 - prob))

        st.write("**Model Matematis:**  z = β₀ + β·x")
        st.write(f"Nilai z = `{z:.4f}`")
        st.write(f"Probabilitas (Sigmoid) = `{prob:.4f}`")
        st.write(f"Prediksi Kelas = `{kelas}`")
        st.write(f"Log Loss = `{log_loss:.4f}`")

        st.info(
            "Logistic Regression cocok untuk klasifikasi biner "
            "dan menghasilkan output probabilistik."
        )

    # =====================================================
    # DECISION TREE
    # =====================================================
    elif algo == "Decision Tree":
        st.subheader("Decision Tree")

        st.markdown("""
        **Tahapan Decision Tree:**
        1. Menghitung impurity awal dataset (entropy)
        2. Memilih fitur dan threshold kandidat
        3. Membagi data menjadi node kiri dan kanan
        4. Menghitung Information Gain
        5. Memilih pemisahan terbaik
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
        st.write(f"Threshold = `{threshold:.4f}`")
        st.success(f"Information Gain = `{information_gain:.4f}`")

        st.info(
            "Decision Tree membangun aturan keputusan berbentuk pohon "
            "berdasarkan pemisahan terbaik data."
        )

    # =====================================================
    # RANDOM FOREST
    # =====================================================
    elif algo == "Random Forest":
        st.subheader("Random Forest")

        st.markdown("""
        **Tahapan Random Forest:**
        1. Melakukan bootstrap sampling dari data
        2. Membangun banyak decision tree
        3. Setiap tree melakukan prediksi sendiri
        4. Hasil akhir ditentukan melalui voting mayoritas
        """)

        n = len(df)
        bootstrap_idx = np.random.choice(df.index, size=n, replace=True)

        fake_preds = np.random.choice(df[target_col].unique(), size=7)
        vote = pd.Series(fake_preds).value_counts()

        st.write("Contoh indeks bootstrap:", bootstrap_idx[:10])
        st.write("Prediksi tiap tree:", fake_preds)
        st.success(f"Hasil voting mayoritas: `{vote.idxmax()}`")

        st.info(
            "Random Forest mengurangi overfitting "
            "dengan menggabungkan banyak model."
        )

    # =====================================================
    # SUPPORT VECTOR MACHINE
    # =====================================================
    elif algo == "Support Vector Machine (SVM)":
        st.subheader("Support Vector Machine (SVM)")

        st.markdown("""
        **Tahapan SVM:**
        1. Menentukan hyperplane pemisah antar kelas
        2. Menghitung jarak data terhadap hyperplane
        3. Menentukan kelas berdasarkan sisi hyperplane
        """)

        w = np.ones(len(X_sample)) * 0.5
        b = -0.2

        decision_value = np.dot(w, X_sample) + b
        kelas = 1 if decision_value >= 0 else 0

        st.write("Decision Function:  f(x) = w·x + b")
        st.write(f"Nilai f(x) = `{decision_value:.4f}`")
        st.success(f"Prediksi Kelas = `{kelas}`")

        st.info(
            "SVM efektif untuk data dengan margin pemisah yang jelas."
        )

    # =====================================================
    # CATBOOST
    # =====================================================
    elif algo == "CatBoost":
        st.subheader("CatBoost")

        st.markdown("""
        **Tahapan CatBoost:**
        1. Membuat model awal sederhana
        2. Menghitung error prediksi
        3. Menambahkan model baru untuk memperbaiki error
        4. Menggabungkan prediksi secara bertahap (boosting)
        """)

        initial_prediction = 0.5
        learning_rate = 0.1
        correction = -0.2

        updated_prediction = initial_prediction + learning_rate * correction

        st.write(f"Prediksi Awal = `{initial_prediction}`")
        st.write(f"Koreksi Error = `{correction}`")
        st.write(f"Prediksi Baru = `{updated_prediction:.4f}`")

        st.success("Prediksi diperbarui melalui mekanisme boosting.")

        st.info(
            "CatBoost sangat efektif untuk data tabular "
            "dan stabil terhadap overfitting."
        )

    # =====================================================
    # PENUTUP
    # =====================================================
    st.markdown("---")
    st.info(
        "Penjelasan ini bersifat **konseptual dan edukatif**. "
        "Training dan evaluasi model sebenarnya dilakukan "
        "pada menu **Machine Learning**."
    )
