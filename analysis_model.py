import streamlit as st

def analysis_model_page():

    st.subheader("Analisis Model Klasifikasi")
    st.markdown(
        "Halaman ini menampilkan **tahapan metode dan rumus matematis** "
        "dari algoritma klasifikasi Machine Learning secara **edukatif**."
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

        st.markdown("1. **Penentuan variabel**")
        st.markdown("Menentukan variabel independen dan variabel dependen biner.")

        st.markdown("2. **Menyusun model linear**")
        st.latex(r"z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n")

        st.markdown("3. **Transformasi sigmoid**")
        st.latex(r"P(y=1) = \frac{1}{1 + e^{-z}}")

        st.markdown("4. **Estimasi parameter**")
        st.markdown("Parameter diestimasi menggunakan Maximum Likelihood Estimation (MLE).")

        st.markdown("5. **Fungsi loss (Log Loss)**")
        st.latex(r"L = -\left[y \log(p) + (1-y)\log(1-p)\right]")

        st.markdown("6. **Optimasi model**")
        st.markdown("Parameter diperbarui menggunakan metode optimasi seperti Gradient Descent.")

        st.markdown("7. **Penentuan kelas**")
        st.markdown("- Jika $P(y=1) \ge 0.5$ → kelas 1  \n- Jika $P(y=1) < 0.5$ → kelas 0")

    # =====================================================
    # DECISION TREE
    # =====================================================
    elif algo == "Decision Tree":
        st.subheader("Decision Tree")

        st.markdown("**Tahapan Decision Tree:**")

        st.markdown("1. **Identifikasi dataset**")
        st.markdown("Dataset terdiri dari fitur dan label kelas.")

        st.markdown("2. **Menghitung entropy awal**")
        st.latex(r"H(S) = -\sum_{i=1}^{k} p_i \log_2(p_i)")

        st.markdown("3. **Menentukan kandidat split**")
        st.markdown("Setiap fitur diuji menggunakan nilai threshold tertentu.")

        st.markdown("4. **Menghitung entropy setelah split**")
        st.latex(r"H(S_v) = -\sum p_{iv} \log_2(p_{iv})")

        st.markdown("5. **Menghitung Information Gain**")
        st.latex(r"IG(S,A) = H(S) - \sum_{v \in A} \frac{|S_v|}{|S|} H(S_v)")

        st.markdown("6. **Pemilihan node**")
        st.markdown("Fitur dengan Information Gain terbesar dipilih sebagai node.")

        st.markdown("7. **Pembentukan pohon keputusan**")
        st.markdown("Proses diulang hingga kriteria berhenti terpenuhi.")

        st.markdown("8. **Penentuan kelas**")
        st.markdown("Kelas daun ditentukan berdasarkan mayoritas data.")

    # =====================================================
    # RANDOM FOREST
    # =====================================================
    elif algo == "Random Forest":
        st.subheader("Random Forest")

        st.markdown("**Tahapan Random Forest:**")

        st.markdown("1. **Bootstrap sampling**")
        st.markdown("Pengambilan sampel data secara acak dengan pengembalian.")

        st.markdown("2. **Pemilihan subset fitur**")
        st.markdown("Setiap tree menggunakan sebagian fitur.")

        st.markdown("3. **Pembangunan Decision Tree**")
        st.markdown("Setiap tree dibangun secara independen.")

        st.markdown("4. **Pembentukan ensemble**")
        st.markdown("Kumpulan decision tree membentuk satu model.")

        st.markdown("5. **Prediksi tiap tree**")
        st.latex(r"y_1, y_2, \dots, y_n")

        st.markdown("6. **Voting mayoritas**")
        st.latex(r"\hat{y} = \text{mode}(y_1, y_2, \dots, y_n)")

        st.markdown("7. **Hasil akhir**")
        st.markdown("Kelas dengan suara terbanyak menjadi prediksi akhir.")

    # =====================================================
    # SUPPORT VECTOR MACHINE
    # =====================================================
    elif algo == "Support Vector Machine (SVM)":
        st.subheader("Support Vector Machine (SVM)")

        st.markdown("**Tahapan Support Vector Machine:**")

        st.markdown("1. **Representasi data**")
        st.markdown("Data dipetakan ke ruang fitur.")

        st.markdown("2. **Menentukan hyperplane**")
        st.latex(r"f(x) = w \cdot x + b")

        st.markdown("3. **Memaksimalkan margin**")
        st.latex(r"\min \frac{1}{2} \|w\|^2")

        st.markdown("4. **Constraint klasifikasi**")
        st.latex(r"y_i (w \cdot x_i + b) \ge 1")

        st.markdown("5. **Support vector**")
        st.markdown("Data terdekat dengan hyperplane menjadi support vector.")

        st.markdown("6. **Klasifikasi data baru**")
        st.markdown("- Jika $f(x) \ge 0$ → kelas 1  \n- Jika $f(x) < 0$ → kelas 0")

    # =====================================================
    # CATBOOST
    # =====================================================
    elif algo == "CatBoost":
        st.subheader("CatBoost")

        st.markdown("**Tahapan CatBoost (Gradient Boosting):**")

        st.markdown("1. **Inisialisasi model**")
        st.latex(r"F_0(x) = \arg\min_{\gamma} \sum L(y_i, \gamma)")

        st.markdown("2. **Prediksi awal**")
        st.markdown("Model menghasilkan prediksi awal.")

        st.markdown("3. **Perhitungan residual**")
        st.latex(r"r_i = y_i - \hat{y}_i")

        st.markdown("4. **Pembangunan weak learner**")
        st.markdown("Model sederhana dibangun berdasarkan residual.")

        st.markdown("5. **Pembaruan model**")
        st.latex(r"F_m(x) = F_{m-1}(x) + \eta \cdot r_i")

        st.markdown("6. **Iterasi bertahap**")
        st.markdown("Proses dilakukan hingga iterasi maksimum atau model konvergen.")

        st.markdown("7. **Prediksi akhir**")
        st.markdown("Prediksi diperoleh dari akumulasi seluruh model.")

    st.markdown("---")
    st.info(
        "Halaman ini bersifat **edukatif** dan hanya menampilkan "
        "tahapan metode serta rumus matematis. "
        "Training dan evaluasi model dilakukan pada menu lain."
    )
