def ml_model():
    st.header("Machine Learning – Water Potability")

    # ===============================
    # 1. Load Dataset
    # ===============================
    try:
        df = pd.read_csv("water_potability.csv")
    except:
        st.error("❌ Gagal memuat dataset water_potability.csv")
        st.stop()

    # ===============================
    # 2. Target Check
    # ===============================
    target = "Potability"
    if target not in df.columns:
        st.error("❌ Kolom target 'Potability' tidak ditemukan")
        st.stop()

    # ===============================
    # 3. Missing Value Handling
    # ===============================
    before = df.shape[0]
    df = df.dropna()
    after = df.shape[0]

    st.write(f"📌 Data sebelum drop NA: **{before}**")
    st.write(f"📌 Data setelah drop NA: **{after}**")

    # ===============================
    # 4. Outlier Handling (IQR)
    # ===============================
    numeric_cols = df.drop(columns=[target]).columns

    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1

    df = df[
        ~(
            (df[numeric_cols] < (Q1 - 1.5 * IQR)) |
            (df[numeric_cols] > (Q3 + 1.5 * IQR))
        ).any(axis=1)
    ]

    # ===============================
    # 5. Normalisasi
    # ===============================
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # ===============================
    # 6. Train Test Split
    # ===============================
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ===============================
    # 7. SMOTE
    # ===============================
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    # ===============================
    # 8. Model List
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

    # ===============================
    # 9. Training & Evaluation
    # ===============================
    st.subheader("Evaluasi Model")

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
    # 10. Best Model
    # ===============================
    best_row = result_df.sort_values(
        by="ROC AUC (%)",
        ascending=False
    ).iloc[0]

    best_model_name = best_row["Model"]
    best_model = trained_models[best_model_name]

    st.success(
        f"Model terbaik: **{best_model_name}** "
        f"(ROC AUC = {best_row['ROC AUC (%)']}%)"
    )

    # ===============================
    # 11. Confusion Matrix
    # ===============================
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, best_model.predict(X_test))
    cm_df = pd.DataFrame(
        cm,
        index=["Actual 0 (Tidak Layak)", "Actual 1 (Layak)"],
        columns=["Predicted 0", "Predicted 1"]
    )

    st.dataframe(cm_df)

    # ===============================
    # 12. Insight
    # ===============================
    st.subheader("Insight Model")
    st.markdown("""
    - Model digunakan untuk **prediksi kelayakan air minum**
    - **ROC AUC tinggi** menunjukkan model mampu membedakan air layak & tidak layak
    - Model **bukan pengganti uji laboratorium**, tetapi **alat bantu analisis awal**
    """)

    # ===============================
    # 13. Save Model
    # ===============================
    joblib.dump(best_model, "best_model_water.pkl")
    joblib.dump(X.columns, "model_features.pkl")
    joblib.dump(numeric_cols, "numeric_columns.pkl")
