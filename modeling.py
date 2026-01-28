import streamlit as st
import pandas as pd
import numpy as np

# =========================
# PUSTAKA SCIKIT-LEARN
# =========================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier


def modeling_page():

    # =========================
    # PENGAMAN DATASET
    # =========================
    if "df" not in st.session_state or "dataset_name" not in st.session_state:
        st.warning(
            "Silakan unggah dataset terlebih dahulu "
            "melalui menu Pemilihan Dataset."
        )
        return

    df = st.session_state["df"]
    dataset_name = st.session_state["dataset_name"]

    # =========================
    # JUDUL & DESKRIPSI HALAMAN
    # =========================
    st.subheader("Pemodelan Machine Learning")

    st.markdown("""
    Halaman ini digunakan untuk melakukan **pelatihan (training)** dan
    **evaluasi model klasifikasi** menggunakan alur
    **Machine Learning end-to-end**.

    Tahapan yang dilakukan meliputi:
    - Pra-pemrosesan data
    - Pembagian data latih dan data uji
    - Pelatihan beberapa algoritma klasifikasi
    - Evaluasi kinerja model
    - Pemilihan model terbaik
    """)

    st.markdown("---")

    # =========================
    # PENENTUAN TARGET OTOMATIS
    # =========================
    if dataset_name == "water_potability.csv":
        target_col = "Potability"
        dataset_type = "Lingkungan"
    elif dataset_name == "cardio_train.csv":
        target_col = "cardio"
        dataset_type = "Kesehatan"
    else:
        st.error("Dataset tidak dikenali oleh sistem.")
        return

    st.write(f"**Dataset:** `{dataset_name}` ({dataset_type})")
    st.write(f"**Variabel Target:** `{target_col}`")

    st.session_state["target_col"] = target_col
    st.session_state["dataset_type"] = dataset_type

    st.markdown("---")

    # =========================
    # PRA-PEMROSESAN DATA
    # =========================
    st.subheader("Pra-pemrosesan Data")

    df_model = df.copy()

    for col in df_model.columns:
        df_model[col] = pd.to_numeric(df_model[col], errors="coerce")

    jumlah_awal = len(df_model)
    df_model = df_model.dropna()
    jumlah_akhir = len(df_model)

    st.info(
        f"Pembersihan data selesai: "
        f"{jumlah_awal - jumlah_akhir} baris dihapus "
        f"karena mengandung nilai hilang."
    )

    X = df_model.drop(columns=[target_col])
    y = df_model[target_col]

    # =========================
    # PEMBAGIAN DATA LATIH & UJI
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # =========================
    # STANDARISASI FITUR
    # =========================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    st.session_state["scaler"] = scaler
    st.session_state["feature_columns"] = X.columns.tolist()

    # =========================
    # DEFINISI MODEL
    # =========================
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
        "Support Vector Machine (SVM)": SVC(
            probability=True, random_state=42
        ),
        "CatBoost": CatBoostClassifier(
            iterations=200,
            learning_rate=0.1,
            depth=6,
            random_state=42,
            verbose=0
        )
    }

    results = []
    confusion_matrices = {}

    model_terbaik = None
    nama_model_terbaik = None
    metrik_terbaik = {}
    nilai_f1_terbaik = 0

    # =========================
    # PELATIHAN & EVALUASI MODEL
    # =========================
    st.subheader("Pelatihan dan Evaluasi Model")

    for nama, model in models.items():

        if nama in ["Logistic Regression", "Support Vector Machine (SVM)"]:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        confusion_matrices[nama] = confusion_matrix(y_test, y_pred)

        results.append({
            "Algoritma": nama,
            "Akurasi (%)": round(acc * 100, 2),
            "Presisi (%)": round(prec * 100, 2),
            "Recall (%)": round(rec * 100, 2),
            "F1-Score (%)": round(f1 * 100, 2)
        })

        if f1 > nilai_f1_terbaik:
            nilai_f1_terbaik = f1
            model_terbaik = model
            nama_model_terbaik = nama
            metrik_terbaik = {
                "akurasi": acc * 100,
                "presisi": prec * 100,
                "recall": rec * 100,
                "f1": f1 * 100
            }

    hasil_df = pd.DataFrame(results)

    # =========================
    # HASIL EVALUASI MODEL
    # =========================
    st.subheader("Hasil Evaluasi Model")
    st.dataframe(hasil_df, use_container_width=True)

    st.success(
        f"""
        **Model Terbaik: {nama_model_terbaik}**

        - Akurasi   : {metrik_terbaik['akurasi']:.2f}%
        - Presisi   : {metrik_terbaik['presisi']:.2f}%
        - Recall    : {metrik_terbaik['recall']:.2f}%
        - F1-Score  : {metrik_terbaik['f1']:.2f}%
        """
    )

    # =========================
    # CONFUSION MATRIX
    # =========================
    st.subheader("Confusion Matrix")

    selected_model = st.selectbox(
        "Pilih model untuk melihat Confusion Matrix",
        list(confusion_matrices.keys())
    )

    cm = confusion_matrices[selected_model]
    tn, fp, fn, tp = cm.ravel()

    cm_labeled = pd.DataFrame(
        [
            [tp, fp],
            [fn, tn]
        ],
        index=[
            "Prediksi Positif (1)",
            "Prediksi Negatif (0)"
        ],
        columns=[
            "Aktual Positif (1)",
            "Aktual Negatif (0)"
        ]
    )

    st.write(f"Confusion Matrix — **{selected_model}**")
    st.dataframe(cm_labeled, use_container_width=True)

    # =========================
    # PENJELASAN CONFUSION MATRIX
    # =========================
    st.markdown("### Penjelasan Confusion Matrix")

    st.markdown(f"""
    - **True Positive (TP)** = {tp}  
      Model memprediksi positif dan data memang positif.

    - **False Positive (FP)** = {fp}  
      Model memprediksi positif, tetapi data sebenarnya negatif.

    - **False Negative (FN)** = {fn}  
      Model memprediksi negatif, tetapi data sebenarnya positif.

    - **True Negative (TN)** = {tn}  
      Model memprediksi negatif dan data memang negatif.
    """)

    st.info(
        "Baris menunjukkan **hasil prediksi model**, "
        "sedangkan kolom menunjukkan **kondisi aktual data**."
    )

    # =========================
    # PENYIMPANAN MODEL TERBAIK
    # =========================
    st.session_state["best_model"] = model_terbaik

    st.info(
        "Model terbaik telah disimpan dan akan digunakan "
        "pada menu **Prediksi**."
    )
