import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE
import joblib

# ===============================
# SAFE CATBOOST IMPORT
# ===============================
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ModuleNotFoundError:
    CATBOOST_AVAILABLE = False


def ml_model():
    st.header("🤖 Machine Learning – Cardiovascular Disease")

    # ===============================
    # 1. Load Dataset (SAFE)
    # ===============================
    try:
        df = pd.read_csv("cardio_train.csv", sep=";")
    except Exception as e:
        st.error("❌ Gagal memuat dataset cardio_train.csv")
        st.stop()

    # Drop ID jika ada
    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)

    # Cek target
    target = "cardio"
    if target not in df.columns:
        st.error("❌ Kolom target 'cardio' tidak ditemukan")
        st.stop()

    # ===============================
    # 2. Outlier Handling (IQR)
    # ===============================
    numeric_cols = (
        df.drop(columns=[target])
        .select_dtypes(include="number")
        .columns
    )

    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1

    before = df.shape[0]
    df = df[
        ~(
            (df[numeric_cols] < (Q1 - 1.5 * IQR)) |
            (df[numeric_cols] > (Q3 + 1.5 * IQR))
        ).any(axis=1)
    ]
    after = df.shape[0]

    st.write(f"📌 Data sebelum outlier removal: **{before}**")
    st.write(f"📌 Data setelah outlier removal: **{after}**")

    # ===============================
    # 3. Encoding
    # ===============================
    df_model = pd.get_dummies(df, drop_first=True)

    # ===============================
    # 4. Normalisasi
    # ===============================
    scaler = MinMaxScaler()
    df_model[numeric_cols] = scaler.fit_transform(df_model[numeric_cols])

    # ===============================
    # 5. Train Test Split
    # ===============================
    X = df_model.drop(columns=[target])
    y = df_model[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ===============================
    # 6. SMOTE
    # ===============================
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    # ===============================
    # 7. Model List
    # ===============================
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),
        "SVM": SVC(probability=True, random_state=42)
    }

    if CATBOOST_AVAILABLE:
        models["CatBoost"] = CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            loss_function="Logloss",
            verbose=False,
            random_state=42
        )
    else:
        st.warning("⚠️ CatBoost tidak tersedia. Model lain tetap dijalankan.")

    # ===============================
    # 8. Training & Evaluation
    # ===============================
    st.subheader("📊 Evaluasi Model")

    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train_bal, y_train_bal)
        trained_models[name] = model

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results.append({
            "Model": name,
            "Accuracy (%)": round(accuracy_score(y_test, y_pred) * 100, 2),
            "Precision (%)": round(precision_score(y_test, y_pred) * 100, 2),
            "Recall (%)": round(recall_score(y_test, y_pred) * 100, 2),
            "F1 Score (%)": round(f1_score(y_test, y_pred) * 100, 2),
            "ROC AUC (%)": round(roc_auc_score(y_test, y_prob) * 100, 2)
        })

    result_df = pd.DataFrame(results)
    st.dataframe(result_df, use_container_width=True)

    # ===============================
    # 9. Best Model
    # ===============================
    best_row = result_df.sort_values(
        by="ROC AUC (%)",
        ascending=False
    ).iloc[0]

    best_model_name = best_row["Model"]
    best_model = trained_models[best_model_name]

    st.success(
        f"🏆 Model terbaik: **{best_model_name}** "
        f"(ROC AUC = {best_row['ROC AUC (%)']}%)"
    )

    # ===============================
    # 10. Confusion Matrix
    # ===============================
    st.subheader("🧮 Confusion Matrix")

    cm = confusion_matrix(y_test, best_model.predict(X_test))
    cm_df = pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"]
    )

    st.dataframe(cm_df)

    # ===============================
    # 11. Insight
    # ===============================
    st.subheader("📝 Insight Model")
    st.markdown("""
    - Model digunakan untuk **prediksi risiko penyakit kardiovaskular**
    - Nilai **ROC AUC tinggi** menandakan performa klasifikasi yang baik
    - Model **bukan alat diagnosis**, tetapi **alat bantu skrining awal**
    """)

    # ===============================
    # 12. Save Model
    # ===============================
    joblib.dump(best_model, "best_model_cardio.pkl")
    joblib.dump(X.columns, "model_features.pkl")
    joblib.dump(numeric_cols, "numeric_columns.pkl")
