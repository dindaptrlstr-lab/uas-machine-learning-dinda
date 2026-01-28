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
        st.warning("Silakan pilih dataset terlebih dahulu pada menu Upload Dataset.")
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

    st.subheader("Machine Learning – Training & Evaluasi Model")

    st.markdown("""
    Halaman ini menampilkan **proses pelatihan dan evaluasi model klasifikasi**,
    mulai dari preprocessing hingga analisis performa menggunakan
    **confusion matrix dan metrik evaluasi klasifikasi**.
    """)

    st.markdown("---")

    # =========================
    # PREPROCESSING
    # =========================
    df_model = df.copy()

    for col in df_model.columns:
        df_model[col] = pd.to_numeric(df_model[col], errors="coerce")

    before = len(df_model)
    df_model = df_model.dropna()
    after = len(df_model)

    st.info(f"Preprocessing: **{before - after}** baris dihapus karena missing value.")

    X = df_model.drop(columns=[target_col])
    y = df_model[target_col]

    # =========================
    # SPLIT DATA
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # =========================
    # SCALING
    # =========================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    st.session_state["scaler"] = scaler
    st.session_state["feature_columns"] = X.columns.tolist()

    # =========================
    # MODEL
    # =========================
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Support Vector Machine (SVM)": SVC(probability=True, random_state=42),
        "CatBoost": CatBoostClassifier(iterations=200, learning_rate=0.1, depth=6, verbose=0)
    }

    results = []
    conf_matrices = {}

    best_model = None
    best_model_name = None
    best_f1 = 0

    # =========================
    # TRAIN & EVALUATE
    # =========================
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

        cm = confusion_matrix(y_test, y_pred)
        conf_matrices[name] = cm

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

    # =========================
    # TABEL EVALUASI
    # =========================
    st.subheader("Hasil Evaluasi Model")
    st.dataframe(pd.DataFrame(results), use_container_width=True)

    st.success(
        f"Model terbaik: **{best_model_name}** "
        f"(F1-Score = {best_f1:.4f})"
    )

    # =========================
    # CONFUSION MATRIX DETAIL
    # =========================
    st.subheader("Confusion Matrix (Detail Interpretasi)")

    selected_model = st.selectbox(
        "Pilih model",
        list(conf_matrices.keys())
    )

    cm = conf_matrices[selected_model]
    TN, FP, FN, TP = cm.ravel()

    cm_df = pd.DataFrame(
        [
            ["Predicted Positive (1)", TP, FP],
            ["Predicted Negative (0)", FN, TN]
        ],
        columns=[
            "Prediksi \\ Aktual",
            "Actual Positive (1)",
            "Actual Negative (0)"
        ]
    )

    st.dataframe(cm_df, use_container_width=True)

    st.markdown("""
    **Keterangan Confusion Matrix:**

    - **True Positive (TP)**  
      Model memprediksi **positif**, dan kondisi sebenarnya **positif**

    - **False Positive (FP)**  
      Model memprediksi **positif**, tetapi kondisi sebenarnya **negatif**

    - **False Negative (FN)**  
      Model memprediksi **negatif**, tetapi kondisi sebenarnya **positif**

    - **True Negative (TN)**  
      Model memprediksi **negatif**, dan kondisi sebenarnya **negatif**
    """)

    st.markdown("""
    **Arah Pembacaan Tabel:**
    - **Horizontal (kolom)** → Kondisi sebenarnya (*Actual*)
    - **Vertikal (baris)** → Hasil prediksi model (*Predicted*)
    """)

    # =========================
    # SIMPAN MODEL
    # =========================
    st.session_state["best_model"] = best_model

    st.info(
        "Model terbaik disimpan dan akan digunakan "
        "pada menu **Prediction App**."
    )
