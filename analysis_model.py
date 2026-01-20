import streamlit as st
import pandas as pd
import numpy as np
from scipy.special import expit  # sigmoid

def analysis_model_page():

    # =========================
    # PENGAMAN DATASET
    # =========================
    if "df" not in st.session_state or "dataset_name" not in st.session_state:
        st.warning("Silakan upload dataset terlebih dahulu.")
        return

    df = st.session_state["df"]
    dataset_name = st.session_state["dataset_name"]

    # =========================
    # TARGET OTOMATIS
    # =========================
    if dataset_name == "water_potability.csv":
        target_col = "Potability"
        dataset_type = "Lingkungan"
    elif dataset_name == "cardio_train.csv":
        target_col = "cardio"
        dataset_type = "Kesehatan"
    else:
        st.error("Dataset tidak dikenali.")
        return

    st.title("🧠 Analisis Model Klasifikasi (Detail Perhitungan)")
    st.write(
        f"Dataset: **{dataset_name}**  \n"
        "Halaman ini menampilkan **logika dan perhitungan inti** dari setiap algoritma klasifikasi."
    )

    st.markdown("---")

    algo = st.selectbox(
        "Pilih Algoritma",
        [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "Support Vector Machine (SVM)",
            "CatBoost"
        ]
    )

    st.markdown("---")

    numeric_df = df.select_dtypes(include="number").drop(columns=[target_col], errors="ignore")
    X_sample = numeric_df.iloc[0].values

    # =========================================================
    # LOGISTIC REGRESSION
    # =========================================================
    if algo == "Logistic Regression":
        st.subheader("📌 Logistic Regression")

        beta = np.ones(len(X_sample)) * 0.1
        beta_0 = 0.1

        z = beta_0 + np.dot(X_sample, beta)
        prob = expit(z)
        kelas = 1 if prob >= 0.5 else 0
        log_loss = -(np.log(prob) if kelas == 1 else np.log(1 - prob))

        st.write("**Rumus:**  z = β₀ + β·x")
        st.write(f"z = `{z:.4f}`")
        st.write(f"Sigmoid(z) = `{prob:.4f}`")
        st.write(f"Prediksi Kelas = `{kelas}`")
        st.write(f"Log Loss (1 data) = `{log_loss:.4f}`")

    # =========================================================
    # DECISION TREE
    # =========================================================
    elif algo == "Decision Tree":
        st.subheader("📌 Decision Tree")

        class_prob = df[target_col].value_counts(normalize=True)
        entropy = -(class_prob * np.log2(class_prob)).sum()

        st.write("**Entropy Awal Dataset:**")
        st.write(f"`{entropy:.4f}`")

        feature = numeric_df.columns[0]
        threshold = df[feature].median()

        left = df[df[feature] <= threshold]
        right = df[df[feature] > threshold]

        def entropy_sub(data):
            p = data[target_col].value_counts(normalize=True)
            return -(p * np.log2(p)).sum()

        ig = entropy - (
            (len(left)/len(df)) * entropy_sub(left) +
            (len(right)/len(df)) * entropy_sub(right)
        )

        st.write(f"**Fitur Contoh:** `{feature}`")
        st.write(f"**Threshold (Median):** `{threshold:.4f}`")
        st.success(f"**Information Gain:** `{ig:.4f}`")

    # =========================================================
    # RANDOM FOREST
    # =========================================================
    elif algo == "Random Forest":
        st.subheader("📌 Random Forest")

        n = len(df)
        bootstrap_idx = np.random.choice(df.index, size=n, replace=True)

        st.write("**Bootstrap Sampling:**")
        st.write("10 indeks pertama:", bootstrap_idx[:10])

        fake_tree_pred = np.random.choice(df[target_col].unique(), size=7)
        vote = pd.Series(fake_tree_pred).value_counts()

        st.write("**Prediksi Tiap Tree:**")
        st.write(fake_tree_pred)

        st.success(f"Hasil Voting Mayoritas: **{vote.idxmax()}**")

    # =========================================================
    # SVM
    # =========================================================
    elif algo == "Support Vector Machine (SVM)":
        st.subheader("📌 Support Vector Machine (SVM)")

        w = np.ones(len(X_sample)) * 0.5
        b = -0.2

        decision_value = np.dot(w, X_sample) + b
        kelas = 1 if decision_value >= 0 else 0

        st.write("**Decision Function:**  f(x) = w·x + b")
        st.write(f"f(x) = `{decision_value:.4f}`")
        st.success(f"Hasil Klasifikasi: **{kelas}**")

    # =========================================================
    # CATBOOST
    # =========================================================
    elif algo == "CatBoost":
        st.subheader("📌 CatBoost")

        st.write("""
        CatBoost adalah **ensemble Gradient Boosting**
        yang membangun pohon keputusan secara bertahap
        untuk memperbaiki error sebelumnya.
        """)

        initial_pred = 0.5
        learning_rate = 0.1
        correction = -0.2

        new_pred = initial_pred + learning_rate * correction

        st.write(f"Prediksi Awal: `{initial_pred}`")
        st.write(f"Koreksi Error: `{correction}`")
        st.write(f"Prediksi Baru: `{new_pred:.4f}`")

        st.success("Prediksi diperbarui menggunakan boosting")

    st.markdown("---")
    st.info(
        "Perhitungan di atas bertujuan **edukatif** untuk memahami "
        "mekanisme internal algoritma. Training dan evaluasi model "
        "dilakukan pada menu **Machine Learning**."
    )
