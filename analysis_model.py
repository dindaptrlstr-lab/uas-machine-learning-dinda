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
    secara **konseptual dan matematis**. Perhitungan yang ditampilkan **bukan hasil training aktual model**,
    melainkan simulasi sederhana untuk membantu pemahaman logika algoritma. Proses pelatihan dan evaluasi model secara nyata
    dilakukan pada menu **Machine Learning**.
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

    # Ambil satu observasi untuk ilustrasi perhitungan
    X_sample = numeric_df.iloc[0].values

    # =====================================================
    # LOGISTIC REGRESSION
    # =====================================================
    if algo == "Logistic Regression":
        st.subheader("Logistic Regression")

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

        st.info(
            "Logistic Regression mengubah kombinasi linear fitur "
            "menjadi probabilitas menggunakan fungsi sigmoid."
        )

    # =====================================================
    # DECISION TREE
    # =====================================================
    elif algo == "Decision Tree":
        st.subheader("Decision Tree")

        class_prob = df[target_col].value_counts(normalize=True)
        entropy = -(class_prob * np.log2(class_prob)).sum()

        st.write("**Entropy Awal Dataset:**")
        st.write(f"`{entropy:.4f}`")

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

        st.write(f"**Fitur Contoh:** `{feature}`")
        st.write(f"**Threshold (Median):** `{threshold:.4f}`")
        st.success(f"**Information Gain:** `{information_gain:.4f}`")

        st.info(
            "Decision Tree memilih fitur pemisah terbaik "
            "berdasarkan nilai Information Gain."
        )

    # =====================================================
    # RANDOM FOREST
    # =====================================================
    elif algo == "Random Forest":
        st.subheader("Random Forest")

        n = len(df)
        bootstrap_idx = np.random.choice(df.index, size=n, replace=True)

        st.write("**Bootstrap Sampling (10 indeks pertama):**")
        st.write(bootstrap_idx[:10])

        fake_tree_predictions = np.random.choice(
            df[target_col].unique(), size=7
        )

        vote_result = pd.Series(fake_tree_predictions).value_counts()

        st.write("**Prediksi dari Tiap Tree:**")
        st.write(fake_tree_predictions)

        st.success(f"Hasil Voting Mayoritas: **{vote_result.idxmax()}**")

        st.info(
            "Random Forest menggabungkan banyak pohon keputusan "
            "untuk meningkatkan stabilitas dan akurasi prediksi."
        )

    # =====================================================
    # SUPPORT VECTOR MACHINE
    # =====================================================
    elif algo == "Support Vector Machine (SVM)":
        st.subheader("Support Vector Machine (SVM)")

        w = np.ones(len(X_sample)) * 0.5
        b = -0.2

        decision_value = np.dot(w, X_sample) + b
        kelas = 1 if decision_value >= 0 else 0

        st.write("**Decision Function:**  f(x) = w·x + b")
        st.write(f"f(x) = `{decision_value:.4f}`")
        st.success(f"Hasil Klasifikasi: **{kelas}**")

        st.info(
            "SVM mengklasifikasikan data berdasarkan posisi relatif "
            "terhadap hyperplane pemisah."
        )

    # =====================================================
    # CATBOOST
    # =====================================================
    elif algo == "CatBoost":
        st.subheader("CatBoost")

        st.markdown("""
        **CatBoost** adalah algoritma **Gradient Boosting**
        yang membangun model secara bertahap,
        dengan setiap model baru berfokus
        memperbaiki kesalahan model sebelumnya.
        """)

        initial_prediction = 0.5
        learning_rate = 0.1
        error_correction = -0.2

        updated_prediction = initial_prediction + learning_rate * error_correction

        st.write(f"Prediksi Awal: `{initial_prediction}`")
        st.write(f"Koreksi Error: `{error_correction}`")
        st.write(f"Prediksi Baru: `{updated_prediction:.4f}`")

        st.success("Prediksi diperbarui menggunakan mekanisme boosting.")

        st.info(
            "CatBoost efektif untuk data tabular "
            "dan membantu mengurangi overfitting."
        )

    # =====================================================
    # CATATAN PENUTUP
    # =====================================================
    st.markdown("---")
    st.info(
        "Perhitungan di halaman ini bersifat **edukatif** "
        "untuk memahami mekanisme algoritma. "
        "Pelatihan dan evaluasi model sebenarnya "
        "dilakukan pada menu **Machine Learning**."
    )


