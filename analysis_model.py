import streamlit as st

def analysis_model_page():

    st.subheader("Analisis Model Klasifikasi")
    st.markdown(
        "Halaman ini menampilkan **tahapan metode, rumus matematis, "
        "dan keterangan cara membaca rumus** dari algoritma klasifikasi "
        "Machine Learning secara **edukatif**."
    )

    st.markdown("---")

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
    # LOGISTIC REGRESSION
    # =====================================================
    if algo == "Logistic Regression":
        st.subheader("Logistic Regression")

        st.markdown("**Tahapan Logistic Regression:**")

        st.markdown("1. **Menyusun model linear**")
        st.latex(r"z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n")
        st.markdown("""
        **Keterangan cara baca:**
        - \(z\) : nilai linear (skor prediksi)
        - \(\beta_0\) : bias / intercept
        - \(\beta_i\) : koefisien fitur ke-\(i\)
        - \(x_i\) : nilai fitur ke-\(i\)
        """)

        st.markdown("2. **Transformasi sigmoid**")
        st.latex(r"P(y=1) = \frac{1}{1 + e^{-z}}")
        st.markdown("""
        **Keterangan cara baca:**
        - \(P(y=1)\) : probabilitas data termasuk kelas 1
        - \(e\) : bilangan eksponensial
        - \(z\) : hasil model linear
        """)

        st.markdown("3. **Fungsi loss (Log Loss)**")
        st.latex(r"L = -\left[y \log(p) + (1-y)\log(1-p)\right]")
        st.markdown("""
        **Keterangan cara baca:**
        - \(L\) : nilai loss
        - \(y\) : label aktual
        - \(p\) : probabilitas hasil prediksi
        """)

        st.markdown("4. **Penentuan kelas**")
        st.markdown("- Jika \(P(y=1) \ge 0.5\) → kelas 1  \n- Jika \(P(y=1) < 0.5\) → kelas 0")

    # =====================================================
    # DECISION TREE
    # =====================================================
    elif algo == "Decision Tree":
        st.subheader("Decision Tree")

        st.markdown("**Tahapan Decision Tree:**")

        st.markdown("1. **Menghitung entropy**")
        st.latex(r"H(S) = -\sum_{i=1}^{k} p_i \log_2(p_i)")
        st.markdown("""
        **Keterangan cara baca:**
        - \(H(S)\) : entropy dataset
        - \(p_i\) : proporsi kelas ke-\(i\)
        - \(k\) : jumlah kelas
        """)

        st.markdown("2. **Information Gain**")
        st.latex(r"IG(S,A) = H(S) - \sum_{v \in A} \frac{|S_v|}{|S|} H(S_v)")
        st.markdown("""
        **Keterangan cara baca:**
        - \(IG(S,A)\) : information gain fitur \(A\)
        - \(S_v\) : subset data hasil split
        - \(|S|\) : jumlah data
        """)

    # =====================================================
    # RANDOM FOREST
    # =====================================================
    elif algo == "Random Forest":
        st.subheader("Random Forest")

        st.markdown("**Tahapan Random Forest:**")

        st.markdown("1. **Voting mayoritas**")
        st.latex(r"\hat{y} = \text{mode}(y_1, y_2, \dots, y_n)")
        st.markdown("""
        **Keterangan cara baca:**
        - \(\hat{y}\) : prediksi akhir
        - \(y_i\) : prediksi dari tree ke-\(i\)
        - \(\text{mode}\) : nilai yang paling sering muncul
        """)

    # =====================================================
    # SUPPORT VECTOR MACHINE
    # =====================================================
    elif algo == "Support Vector Machine (SVM)":
        st.subheader("Support Vector Machine (SVM)")

        st.markdown("**Tahapan SVM:**")

        st.markdown("1. **Fungsi hyperplane**")
        st.latex(r"f(x) = w \cdot x + b")
        st.markdown("""
        **Keterangan cara baca:**
        - \(w\) : vektor bobot
        - \(x\) : vektor fitur
        - \(b\) : bias
        """)

        st.markdown("2. **Fungsi optimasi margin**")
        st.latex(r"\min \frac{1}{2} \|w\|^2")
        st.markdown("""
        **Keterangan cara baca:**
        - \(\|w\|\) : norma vektor bobot
        - Tujuan: memaksimalkan margin pemisah
        """)

    # =====================================================
    # CATBOOST
    # =====================================================
    elif algo == "CatBoost":
        st.subheader("CatBoost")

        st.markdown("**Tahapan CatBoost (Gradient Boosting):**")

        st.markdown("1. **Inisialisasi model**")
        st.latex(r"F_0(x) = \arg\min_{\gamma} \sum L(y_i, \gamma)")
        st.markdown("""
        **Keterangan cara baca:**
        - \(F_0(x)\) : model awal
        - \(\gamma\) : parameter model
        - \(L\) : fungsi loss
        """)

        st.markdown("2. **Perhitungan residual**")
        st.latex(r"r_i = y_i - \hat{y}_i")
        st.markdown("""
        **Keterangan cara baca:**
        - \(r_i\) : residual data ke-\(i\)
        - \(y_i\) : nilai aktual
        - \(\hat{y}_i\) : nilai prediksi
        """)

        st.markdown("3. **Pembaruan model**")
        st.latex(r"F_m(x) = F_{m-1}(x) + \eta \cdot r_i")
        st.markdown("""
        **Keterangan cara baca:**
        - \(F_m(x)\) : model iterasi ke-\(m\)
        - \(\eta\) : learning rate
        """)

    st.markdown("---")
    st.info(
        "Seluruh rumus disertai **cara baca dan arti simbol** "
        "untuk memudahkan pemahaman konsep."
    )
