import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from catboost import CatBoostClassifier


def modeling_page():

    # =========================
    # PENGAMAN
    # =========================
    if "df" not in st.session_state or "dataset_name" not in st.session_state:
        st.warning("Silakan upload dataset terlebih dahulu.")
        return

    df = st.session_state["df"]
    dataset_name = st.session_state["dataset_name"]

    st.title("🤖 Machine Learning")
    st.write(
        "Halaman ini digunakan untuk **melatih dan membandingkan model klasifikasi** "
        "menggunakan dataset yang di-upload."
    )

    # =========================
    # TARGET OTOMATIS
    # =========================
    if dataset_name == "water_potability.csv":
        target_col = "Potability"
    elif dataset_name == "cardio_train.csv":
        target_col = "cardio"
    else:
        st.error("Dataset tidak dikenali.")
        return

    st.write(f"**Target Klasifikasi:** `{target_col}`")

    # =========================
    # PREPROCESSING
    # =========================
    df_model = df.copy()

    # Paksa numerik
    for col in df_model.columns:
        df_model[col] = pd.to_numeric(df_model[col], errors="coerce")

    # Hapus missing value
    before = len(df_model)
    df_model = df_model.dropna()
    after = len(df_model)

    st.info(f"🧹 Data dibersihkan: {before - after} baris dibuang")

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
    # SCALING (untuk LR & SVM)
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
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),
        "SVM": SVC(probability=True, random_state=42),
        "CatBoost": CatBoostClassifier(
            iterations=200,
            learning_rate=0.1,
            depth=6,
            verbose=0,
            random_state=42
        )
    }

    results = []
    best_model = None
    best_f1 = 0
    best_model_name = None

    # =========================
    # TRAIN & EVALUATION
    # =========================
    for name, model in models.items():

        # Scaling hanya untuk model tertentu
        if name in ["Logistic Regression", "SVM"]:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results.append({
            "Algoritma": name,
            "Accuracy (%)": round(acc * 100, 2),
            "Precision (%)": round(prec * 100, 2),
            "Recall (%)": round(rec * 100, 2),
            "F1-Score (%)": round(f1 * 100, 2),
        })

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name

    results_df = pd.DataFrame(results)

    # =========================
    # OUTPUT
    # =========================
    st.subheader("📊 Hasil Evaluasi Model")
    st.dataframe(results_df, use_container_width=True)

    st.success(
        f"🏆 Model Terbaik: **{best_model_name}** "
        f"(F1-Score = {best_f1:.4f})"
    )

    st.session_state["best_model"] = best_model

    st.info(
        "Model terbaik disimpan dan akan digunakan "
        "pada menu **Prediction App**."
    )
