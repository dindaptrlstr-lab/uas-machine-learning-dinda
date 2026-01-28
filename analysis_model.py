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
    st.subheader("Analisis Model Klasifikasi (Tahapan & Rumus)")

    st.write(
        f"Dataset: **{dataset_name}** ({dataset_type})  \n"
        "Halaman ini menjelaskan **tahapan kerja dan rumus matematis** "
        "dari setiap algoritma klasifikasi."
    )

    st.info(
        "Perhitungan pada halaman ini bersifat **simulasi edukatif**, "
        "bukan hasil training aktual. Training sesungguhnya dilakukan "
        "pada menu **Machine Learning**."
    )

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
    # SAMPLE DATA
    # =====================================================
    numeric_df = df.select_dtypes(include="number").drop(
        columns=[target_col], errors="ignore"
    )

    X_sample = numeric_df.iloc[0].values

    # =====================================================
    # LOGISTIC REGRESSION
    # =====================================================
    if algo == "Logistic Regression":
        st.subheader("Logistic Regression")

        st.markdown("""
        ### Tahapan Logistic Regression
        1. Menghitung kombinasi linear fitur  
           \\[
           z = \\beta_0 + \\beta_1x_1 + \\beta_2x_2 + ... + \\beta_nx_n
           \\]
        2. Mengubah nilai `z` menjadi probabilitas dengan fungsi sigmoid  
           \\[
           P(y=1) = \\frac{1}{1 + e^{-z}}
           \\]
        3. Menentukan kelas berdasarkan threshold (umumnya 0.5)
        4. Menghitung kesalahan prediksi menggunakan **Log Loss**
        """)

        beta = np.ones(len(X_sample)) * 0.1
        beta_0 = 0.1

        z = beta_0 + np.dot(X_sample, beta)
        prob = expit(z)
        kelas = 1 if prob >= 0.5 else 0
        log_loss = -(np.log(prob) if kelas == 1 else np.log(1 - prob))

        st.markdown("### Contoh Perhitungan")
        st.write(f"z = `{z:.4f}`")
        st.write(f"Sigmoid(z) = `{prob:.4f}`")
        st.write(f"Prediksi Kelas = `{kelas}`")
        st.write(f"Log Loss = `{log_loss:.4f}`")

    # =====================================================
    # DECISION TREE
    # =====================================================
    elif algo == "Decision Tree":
        st.subheader("Decision Tree")

        st.markdown("""
        ### Tahapan Decision Tree
        1. Menghitung **entropy awal** dataset  
           \\[
           Entropy = - \\sum p_i \\log_2(p_i)
           \\]
        2. Memilih fitur dan threshold terbaik
        3. Menghitung **Information Gain**  
           \\[
           IG = Entropy_{awal} - Entropy_{setelah\ split}
           \\]
        4. Membuat cabang hingga kondisi berhenti terpenuhi
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

        ig = entropy - (
            (len(left)/len(df))*entropy_subset(left) +
            (len(right)/len(df))*entropy_subset(right)
        )

        st.markdown("### Contoh Perhitungan")
        st.write(f"Entropy Awal = `{entropy:.4f}`")
        st.write(f"Fitur = `{feature}`")
        st.write(f"Information Gain = `{ig:.4f}`")

    # =====================================================
    # RANDOM FOREST
    # =====================================================
    elif algo == "Random Forest":
        st.subheader("Random Forest")

        st.markdown("""
        ### Tahapan Random Forest
        1. Melakukan **bootstrap sampling** pada data
        2. Membangun banyak decision tree
        3. Setiap tree dilatih dengan subset fitur acak
        4. Prediksi akhir ditentukan dengan **majority voting**
        """)

        fake_preds = np.random.choice(df[target_col].unique(), size=7)
        vote = pd.Series(fake_preds).value_counts().idxmax()

        st.markdown("### Contoh Voting")
        st.write("Prediksi tiap tree:", fake_preds)
        st.success(f"Hasil Voting Mayoritas = {vote}")

    # =====================================================
    # SUPPORT VECTOR MACHINE
    # =====================================================
    elif algo == "Support Vector Machine (SVM)":
        st.subheader("Support Vector Machine (SVM)")

        st.markdown("""
        ### Tahapan SVM
        1. Menentukan hyperplane pemisah  
           \\[
           f(x) = w \\cdot x + b
           \\]
        2. Memaksimalkan margin antar kelas
        3. Data diklasifikasikan berdasarkan posisi terhadap hyperplane
        """)

        w = np.ones(len(X_sample)) * 0.5
        b = -0.2

        fx = np.dot(w, X_sample) + b
        kelas = 1 if fx >= 0 else 0

        st.markdown("### Contoh Perhitungan")
        st.write(f"f(x) = `{fx:.4f}`")
        st.success(f"Prediksi Kelas = {kelas}")

    # =====================================================
    # CATBOOST
    # =====================================================
    elif algo == "CatBoost":
        st.subheader("CatBoost")

        st.markdown("""
        ### Tahapan CatBoost
        1. Memulai dari prediksi awal (baseline)
        2. Menghitung error residual
        3. Menambahkan tree baru untuk memperbaiki error
        4. Update prediksi dengan learning rate  
           \\[
           F_{baru} = F_{lama} + \\eta \\times error
           \\]
        """)

        initial_pred = 0.5
        learning_rate = 0.1
        error = -0.2

        new_pred = initial_pred + learning_rate * error

        st.markdown("### Contoh Perhitungan")
        st.write(f"Prediksi Awal = `{initial_pred}`")
        st.write(f"Prediksi Baru = `{new_pred:.4f}`")

    # =====================================================
    # PENUTUP
    # =====================================================
    st.markdown("---")
    st.info(
        "Tahapan dan rumus di atas digunakan untuk **memahami konsep algoritma**. "
        "Evaluasi performa model sesungguhnya dilakukan pada menu **Machine Learning**."
    )
