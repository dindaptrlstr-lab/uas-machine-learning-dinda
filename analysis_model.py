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
    Halaman ini bersifat **edukatif**, bertujuan menjelaskan
    **alur kerja dan rumus matematis** dari algoritma Machine Learning.
    Nilai numerik yang ditampilkan merupakan **simulasi sederhana**,
    bukan hasil training aktual.
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

        1. **Menyusun model linear**
        \[
        z = \\beta_0 + \\beta_1x_1 + \\beta_2x_2 + \\dots + \\beta_nx_n
        \]

        2. **Transformasi sigmoid**
        \[
        P(y=1) = \\frac{1}{1 + e^{-z}}
        \]

        3. **Penentuan kelas**
        - Jika \(P(y=1) \\ge 0.5\) → kelas 1  
        - Jika \(P(y=1) < 0.5\) → kelas 0

        4. **Log Loss**
        \[
        L = -[y\\log(p) + (1-y)\\log(1-p)]
        \]
        """)

        beta = np.ones(len(X_sample)) * 0.1
        beta_0 = 0.1

        z = beta_0 + np.dot(X_sample, beta)
        prob = expit(z)
        kelas = 1 if prob >= 0.5 else 0
        log_loss = -(np.log(prob) if kelas == 1 else np.log(1 - prob))

        st.write("**Model:** z = β₀ + β·x")
        st.write(f"Nilai z = `{z}`")
        st.write(f"Sigmoid(z) = `{prob}`")
        st.write(f"Prediksi Kelas = `{kelas}`")
        st.write(f"Log Loss (1 data) = `{log_loss}`")

    # =====================================================
    # DECISION TREE
    # =====================================================
    elif algo == "Decision Tree":
        st.subheader("Decision Tree")

        st.markdown("""
        **Tahapan Decision Tree:**

        1. **Menghitung Entropy**
        \[
        H(S) = -\\sum p_i \\log_2(p_i)
        \]

        2. **Menguji pemisahan fitur**
        Data dibagi berdasarkan threshold tertentu.

        3. **Information Gain**
        \[
        IG = H(S) - \\sum \\frac{|S_v|}{|S|}H(S_v)
        \]

        Fitur dengan **Information Gain terbesar** dipilih sebagai node.
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

        st.write(f"Entropy Awal = `{entropy}`")
        st.write(f"Fitur = `{feature}`")
        st.write(f"Threshold = `{threshold}`")
        st.write(f"Information Gain = `{information_gain}`")

    # =====================================================
    # RANDOM FOREST
    # =====================================================
    elif algo == "Random Forest":
        st.subheader("Random Forest")

        st.markdown("""
        **Tahapan Random Forest:**

        1. **Bootstrap Sampling**
        Data diambil secara acak dengan pengembalian.

        2. **Pembuatan banyak Decision Tree**
        Setiap tree menggunakan subset data & fitur.

        3. **Voting Mayoritas**
        Hasil prediksi ditentukan berdasarkan suara terbanyak:
        \[
        y = \\text{mode}(y_1, y_2, \\dots, y_n)
        \]
        """)

        n = len(df)
        bootstrap_idx = np.random.choice(df.index, size=n, replace=True)
        fake_tree_predictions = np.random.choice(df[target_col].unique(), size=7)
        vote_result = pd.Series(fake_tree_predictions).value_counts()

        st.write("Contoh prediksi tree:", fake_tree_predictions)
        st.write("Hasil voting:", vote_result.idxmax())

    # =====================================================
    # SUPPORT VECTOR MACHINE
    # =====================================================
    elif algo == "Support Vector Machine (SVM)":
        st.subheader("Support Vector Machine (SVM)")

        st.markdown("""
        **Tahapan SVM:**

        1. **Mencari hyperplane optimal**
        \[
        f(x) = w \\cdot x + b
        \]

        2. **Margin maksimum**
        Hyperplane dipilih agar jarak ke data terdekat maksimum.

        3. **Klasifikasi**
        - Jika \(f(x) \\ge 0\) → kelas 1  
        - Jika \(f(x) < 0\) → kelas 0
        """)

        w = np.ones(len(X_sample)) * 0.5
        b = -0.2

        decision_value = np.dot(w, X_sample) + b
        kelas = 1 if decision_value >= 0 else 0

        st.write(f"f(x) = `{decision_value}`")
        st.write(f"Prediksi Kelas = `{kelas}`")

    # =====================================================
    # CATBOOST
    # =====================================================
    elif algo == "CatBoost":
        st.subheader("CatBoost")

        st.markdown("""
        **Tahapan CatBoost (Gradient Boosting):**

        1. **Prediksi awal**
        \[
        F_0(x)
        \]

        2. **Hitung error**
        \[
        error = y - \\hat{y}
        \]

        3. **Update model**
        \[
        F_m(x) = F_{m-1}(x) + \\eta \\cdot error
        \]

        Proses dilakukan berulang hingga konvergen.
        """)

        initial_prediction = 0.5
        learning_rate = 0.1
        error_correction = -0.2
        updated_prediction = initial_prediction + learning_rate * error_correction

        st.write(f"Prediksi awal = `{initial_prediction}`")
        st.write(f"Prediksi baru = `{updated_prediction}`")

    # =====================================================
    # CATATAN PENUTUP
    # =====================================================
    st.markdown("---")
    st.info(
        "Bagian ini bersifat **edukatif** untuk menjelaskan "
        "tahapan dan rumus algoritma. "
        "Training dan evaluasi aktual dilakukan pada menu **Machine Learning**."
    )
