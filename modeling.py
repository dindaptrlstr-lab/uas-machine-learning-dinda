import streamlit as st
import pandas as pd
import numpy as np

# =========================
# SKLEARN
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
        st.warning("Silakan upload dataset terlebih dahulu melalui sidebar.")
        return

    df = st.session_state["df"]
    dataset_name = st.session_state["dataset_name"]

    # =========================
    # JUDUL & DESKRIPSI HALAMAN
    # =========================
    st.subheader("Machine Learning")

    st.markdown("""
    Halaman ini digunakan untuk melakukan **pelatihan (training)**
    dan **evaluasi model klasifikasi** menggunakan pipeline
    **Machine Learning end-to-end**.

    Proses yang dilakukan meliputi:
    - Preprocessing data
    - Pembagian data latih dan data uji
    - Pelatihan beberapa algoritma klasifikasi
    - Evaluasi performa model
    - Pemilihan model terbaik
    """)

    st.markdown("---")

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

    st.write(f"**Dataset:** `{dataset_name}` ({dataset_type})")
    st.write(f"**Target Klasifikasi:** `{target_col}`")

    # Simpan target & tipe dataset untuk halaman lain
    st.session_state["target_col"] = target_col
    st.session_state["dataset_type"] = dataset_type

    st.markdown("---")

    # =========================
    # PREPROCESSING
    # =========================
    st.subheader("Preprocessing Data")

    df_model = df.copy()

    # Pastikan seluruh kolom numerik
    for col in df_model.columns:
        df_model[col] = pd.to_numeric(df_model[col], errors="coerce")

    before_rows = len(df_model)
    df_model = df_model.dropna()
    after_rows = len(df_model)

    st.info(f"Data dibersihkan: **{before_rows - after_rows}** baris dibuang karena missing value.")

    # Pisahkan fitur & target
    X = df_model.drop(columns=[target_col])
    y = df_model[target_col]

    # =========================
    # TRAIN TEST SPLIT
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # =========================
    # FEATURE SCALING
    # =========================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Simpan scaler & fitur untuk prediksi
    st.session_state["scaler"] = scaler
    st.session_state["feature_columns"] = X.columns.tolist()

    # =========================
    # DEFINISI MODEL
    # =========================
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),
        "Support Vector Machine (SVM)": SVC(
            probability=True,
            random_state=42
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
    conf_matrices = {}

    best_model = None
    best_model_name = None
    best_f1 = 0

    # =========================
    # TRAINING & EVALUATION
    # =========================
    st.subheader("Training & Evaluasi Model")

    for name, model in models.items():

        # Scaling hanya untuk model berbasis jarak / linear
        if name in ["Logistic Regression", "Support Vector Machine (SVM)"]:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        conf_matrices[name] = confusion_matrix(y_test, y_pred)

        results.append({
            "Algoritma": name,
            "Accuracy (%)": round(acc * 100, 2),
            "Precision (%)": round(prec * 100, 2),
            "Recall (%)": round(rec * 100, 2),
            "F1-Score (%)": round(f1 * 100, 2)
        })

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name

    results_df = pd.DataFrame(results)

    # =========================
    # HASIL EVALUASI
    # =========================
    st.subheader("Hasil Evaluasi Model")
    st.dataframe(results_df, use_container_width=True)

    st.success(
        f"Model Terbaik: **{best_model_name}** "
        f"(F1-Score = {best_f1:.4f})"
    )

    # =========================
    # CONFUSION MATRIX
    # =========================
    st.subheader("Confusion Matrix")

    selected_model = st.selectbox(
        "Pilih model untuk melihat confusion matrix",
        list(conf_matrices.keys())
    )

    st.write(f"Confusion Matrix — **{selected_model}**")
    st.dataframe(conf_matrices[selected_model])

    st.markdown("""
    **Penjelasan:**
    - Nilai diagonal menunjukkan prediksi yang benar
    - Nilai di luar diagonal menunjukkan kesalahan klasifikasi
    """)

    # =========================
    # SIMPAN MODEL TERBAIK
    # =========================
    st.session_state["best_model"] = best_model

    st.info(
        "Model terbaik telah disimpan dan "
        "akan digunakan pada menu **Prediction App**."
    )
