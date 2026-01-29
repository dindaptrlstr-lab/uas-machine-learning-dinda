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
        st.warning("Silakan upload dataset terlebih dahulu melalui menu Upload Dataset.")
        return

    df = st.session_state["df"]
    dataset_name = st.session_state["dataset_name"]

    # =========================
    # JUDUL & DESKRIPSI
    # =========================
    st.subheader("Machine Learning")

    st.markdown("""
    Halaman ini digunakan untuk melakukan pelatihan (training) dan
    evaluasi model klasifikasi menggunakan pipeline
    Machine Learning end-to-end.

    Tahapan yang dilakukan:
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

    st.session_state["target_col"] = target_col
    st.session_state["dataset_type"] = dataset_type

    st.markdown("---")

    # =========================
    # PREPROCESSING
    # =========================
    st.subheader("1️⃣ Preprocessing Data")

    df_model = df.copy()
    for col in df_model.columns:
        df_model[col] = pd.to_numeric(df_model[col], errors="coerce")

    before_rows = len(df_model)
    df_model = df_model.dropna()
    after_rows = len(df_model)

    st.info(
        f"Data dibersihkan dari missing value.\n\n"
        f"- Jumlah data awal  : {before_rows}\n"
        f"- Jumlah data akhir : {after_rows}\n"
        f"- Data terhapus     : {before_rows - after_rows}"
    )

    X = df_model.drop(columns=[target_col])
    y = df_model[target_col]

    # =========================
    # TRAIN TEST SPLIT
    # =========================
    st.subheader("2️⃣ Pembagian Data Latih dan Data Uji")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    st.success(
        f"""
        Data dibagi menggunakan **Train-Test Split (80 : 20)**.

        - Jumlah data latih : {len(X_train)}
        - Jumlah data uji   : {len(X_test)}
        """
    )

    # =========================
    # FEATURE SCALING
    # =========================
    st.subheader("3️⃣ Feature Scaling")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    st.info(
        "Standardisasi fitur dilakukan menggunakan **StandardScaler** "
        "agar seluruh fitur berada pada skala yang sebanding."
    )

    st.session_state["scaler"] = scaler
    st.session_state["feature_columns"] = X.columns.tolist()

    # =========================
    # DEFINISI MODEL
    # =========================
    st.subheader("4️⃣ Pelatihan Beberapa Algoritma Klasifikasi")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Support Vector Machine (SVM)": SVC(probability=True, random_state=42),
        "CatBoost": CatBoostClassifier(
            iterations=200,
            learning_rate=0.1,
            depth=6,
            random_state=42,
            verbose=0
        )
    }

    st.write("Algoritma yang digunakan:")
    for m in models.keys():
        st.markdown(f"- {m}")

    results = []
    conf_matrices = {}

    best_model = None
    best_model_name = None
    best_metrics = {}
    best_f1 = 0

    # =========================
    # TRAINING & EVALUATION
    # =========================
    st.subheader("5️⃣ Evaluasi Performa Model")

    for name, model in models.items():

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
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4)
        })

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name
            best_metrics = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1
            }

    results_df = pd.DataFrame(results)

    # =========================
    # HASIL EVALUASI
    # =========================
    st.dataframe(results_df, use_container_width=True)

    st.success(
        f"""
        **Model Terbaik: {best_model_name}**

        - Accuracy  : {best_metrics['accuracy']:.4f}
        - Precision : {best_metrics['precision']:.4f}
        - Recall    : {best_metrics['recall']:.4f}
        - F1-Score  : {best_metrics['f1']:.4f}
        """
    )

    # =========================
    # CONFUSION MATRIX
    # =========================
    st.subheader("Confusion Matrix")

    selected_model = st.selectbox(
        "Pilih model untuk melihat confusion matrix",
        list(conf_matrices.keys())
    )

    cm = conf_matrices[selected_model]
    tn, fp, fn, tp = cm.ravel()

    cm_labeled = pd.DataFrame(
        [[tp, fp], [fn, tn]],
        index=["Prediksi Positif (1)", "Prediksi Negatif (0)"],
        columns=["Aktual Positif (1)", "Aktual Negatif (0)"]
    )

    st.dataframe(cm_labeled, use_container_width=True)

    # =========================
    # SIMPAN MODEL TERBAIK
    # =========================
    st.session_state["best_model"] = best_model

    st.info(
        "Model terbaik telah disimpan dan akan digunakan "
        "pada menu **Prediction App**."
    )

