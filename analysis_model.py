import streamlit as st

def analysis_model_page():

    st.subheader("Analisis Model Klasifikasi (Tahapan Metode & Rumus)")

    st.markdown("""
    Halaman ini menyajikan **tahapan metodologis dan rumus matematis**
    dari algoritma klasifikasi Machine Learning.
    Seluruh penjelasan bersifat **konseptual dan edukatif**.
    """)

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

        st.markdown(r"""
        **Tahapan Logistic Regression:**

        1. **Penentuan variabel**
        Menentukan variabel independen \(x_1, x_2, \dots, x_n\) dan
        variabel dependen \(y \in \{0,1\}\).

        2. **Menyusun model linear**
        \[
        z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n
        \]

        3. **Transformasi sigmoid**
        \[
        P(y=1) = \frac{1}{1 + e^{-z}}
        \]

        4. **Estimasi parameter**
        Parameter \(\beta\) diestimasi menggunakan
        Maximum Likelihood Estimation (MLE).

        5. **Fungsi loss (Log Loss)**
        \[
        L = -\left[y \log(p) + (1-y)\log(1-p)\right]
        \]

        6. **Optimasi**
        Parameter diperbarui menggunakan metode optimasi
        (misalnya Gradient Descent).

        7. **Penentuan kelas**
        - Jika \(P(y=1) \ge 0.5\) → kelas 1  
        - Jika \(P(y=1) < 0.5\) → kelas 0
        """)

    # =====================================================
    # DECISION TREE
    # =====================================================
    elif algo == "Decision Tree":
        st.subheader("Decision Tree")

        st.markdown(r"""
        **Tahapan Decision Tree:**

        1. **Identifikasi dataset**
        Dataset terdiri dari fitur dan label kelas.

        2. **Menghitung entropy awal**
        \[
        H(S) = -\sum_{i=1}^{k} p_i \log_2(p_i)
        \]

        3. **Menentukan kandidat split**
        Setiap fitur diuji menggunakan nilai threshold tertentu.

        4. **Menghitung entropy setelah split**
        \[
        H(S_v) = -\sum p_{iv} \log_2(p_{iv})
        \]

        5. **Menghitung Information Gain**
        \[
        IG(S, A) = H(S) - \sum_{v \in A} \frac{|S_v|}{|S|} H(S_v)
        \]

        6. **Pemilihan node**
        Fitur dengan nilai Information Gain terbesar
        dipilih sebagai node.

        7. **Pembentukan pohon**
        Proses diulang hingga kriteria berhenti terpenuhi
        (pure node, depth maksimum, atau minimum data).

        8. **Penentuan kelas**
        Kelas daun ditentukan berdasarkan mayoritas data.
        """)

    # =====================================================
    # RANDOM FOREST
    # =====================================================
    elif algo == "Random Forest":
        st.subheader("Random Forest")

        st.markdown(r"""
        **Tahapan Random Forest:**

        1. **Bootstrap sampling**
        Pengambilan sampel data secara acak dengan pengembalian.

        2. **Pemilihan subset fitur**
        Setiap tree menggunakan sebagian fitur.

        3. **Pembangunan Decision Tree**
        Setiap tree dibangun secara independen.

        4. **Pembentukan ensemble**
        Kumpulan decision tree membentuk satu model.

        5. **Prediksi tiap tree**
        \[
        y_1, y_2, \dots, y_n
        \]

        6. **Voting mayoritas**
        \[
        \hat{y} = \text{mode}(y_1, y_2, \dots, y_n)
        \]

        7. **Hasil akhir**
        Kelas dengan suara terbanyak menjadi prediksi akhir.
        """)

    # =====================================================
    # SUPPORT VECTOR MACHINE
    # =====================================================
    elif algo == "Support Vector Machine (SVM)":
        st.subheader("Support Vector Machine (SVM)")

        st.markdown(r"""
        **Tahapan Support Vector Machine (SVM):**

        1. **Representasi data**
        Data dipetakan ke ruang fitur.

        2. **Menentukan hyperplane**
        \[
        f(x) = w \cdot x + b
        \]

        3. **Memaksimalkan margin**
        \[
        \min \frac{1}{2} ||w||^2
        \]

        4. **Constraint klasifikasi**
        \[
        y_i (w \cdot x_i + b) \ge 1
        \]

        5. **Support vector**
        Data terdekat dengan hyperplane disebut support vector.

        6. **Klasifikasi data baru**
        - Jika \(f(x) \ge 0\) → kelas 1  
        - Jika \(f(x) < 0\) → kelas 0
        """)

    # =====================================================
    # CATBOOST
    # =====================================================
    elif algo == "CatBoost":
        st.subheader("CatBoost")

        st.markdown(r"""
        **Tahapan CatBoost (Gradient Boosting):**

        1. **Inisialisasi model**
        \[
        F_0(x) = \arg\min_\gamma \sum L(y_i, \gamma)
        \]

        2. **Prediksi awal**
        Model menghasilkan prediksi awal.

        3. **Perhitungan residual**
        \[
        r_i = y_i - \hat{y}_i
        \]

        4. **Pembangunan weak learner**
        Model sederhana dibangun berdasarkan residual.

        5. **Pembaruan model**
        \[
        F_m(x) = F_{m-1}(x) + \eta \cdot r_i
        \]

        6. **Iterasi bertahap**
        Proses dilakukan hingga iterasi maksimum
        atau model konvergen.

        7. **Prediksi akhir**
        Prediksi diperoleh dari akumulasi seluruh model.
        """)

    st.markdown("---")
    st.info(
        "Halaman ini bersifat **edukatif** dan menampilkan "
        "tahapan metode serta rumus matematis. "
        "Training dan evaluasi model dilakukan pada menu lain."
    )
