# Machine Learning Classification with Streamlit

## Latar Belakang
Perkembangan Machine Learning memungkinkan pemanfaatan data dalam berbagai bidang,
termasuk kesehatan dan lingkungan, untuk mendukung pengambilan keputusan berbasis data.
Metode klasifikasi digunakan secara luas untuk memprediksi suatu kondisi atau kategori
berdasarkan karakteristik numerik tertentu.

Proyek ini dikembangkan sebagai implementasi praktis dari konsep Machine Learning
klasifikasi, mulai dari tahap eksplorasi data, pemodelan, evaluasi, hingga deployment
sederhana dalam bentuk aplikasi web interaktif menggunakan Streamlit. Proyek ini
dibuat sebagai bagian dari pembelajaran mata kuliah Machine Learning pada Program
Studi Sains Data.

## Tujuan Proyek
Tujuan dari pengembangan proyek ini adalah:
1. Menerapkan algoritma Machine Learning klasifikasi pada data kesehatan dan lingkungan
2. Melakukan Exploratory Data Analysis (EDA) untuk memahami karakteristik dataset
3. Membandingkan performa beberapa algoritma Machine Learning klasifikasi
4. Menyajikan hasil analisis dan prediksi dalam bentuk aplikasi web interaktif
5. Mengimplementasikan end-to-end pipeline Machine Learning dari data hingga deployment

## Dataset yang Digunakan
Proyek ini menggunakan dua dataset utama dalam format `.csv`, yaitu:

1. Cardio Train Dataset  
   Dataset bidang kesehatan yang berisi data pasien seperti usia, tekanan darah,
   kadar kolesterol, dan indikator kesehatan lainnya untuk memprediksi risiko
   penyakit kardiovaskular.

2. Water Potability Dataset  
   Dataset bidang lingkungan yang digunakan untuk menentukan kelayakan air minum
   berdasarkan parameter fisik dan kimia air.

Sebelum digunakan dalam pemodelan, dataset melalui tahap preprocessing yang meliputi
penanganan missing value, normalisasi data, serta pembagian data menjadi data latih
dan data uji.

## Metode dan Algoritma Machine Learning
Metode yang digunakan dalam proyek ini adalah supervised learning dengan pendekatan
klasifikasi. Algoritma Machine Learning yang diterapkan meliputi:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- CatBoost Classifier

Evaluasi model dilakukan menggunakan metrik evaluasi klasifikasi seperti akurasi,
confusion matrix, precision, recall, dan F1-score untuk menentukan model dengan
performa terbaik. Model terbaik selanjutnya digunakan pada fitur prediksi.

## Fitur Aplikasi
Aplikasi Streamlit yang dikembangkan memiliki fitur-fitur sebagai berikut:
- Dashboard eksplorasi data (EDA)
- Visualisasi distribusi dan karakteristik data
- Training dan evaluasi model Machine Learning
- Perbandingan performa antar model
- Prediksi data baru berdasarkan input pengguna menggunakan model terbaik

## Tools dan Library
Tools dan library yang digunakan dalam proyek ini meliputi:
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- CatBoost

Seluruh dependensi proyek telah tercantum dalam file `requirements.txt`.

## Cara Menjalankan Aplikasi
Untuk menjalankan aplikasi secara lokal, ikuti langkah-langkah berikut:

1. Clone repository:
   git clone https://github.com/dindaptrlstr-lab/uas-machine-learning-dinda.git

2. Masuk ke direktori proyek:
   cd uas-machine-learning-dinda

3. Install dependensi:
   pip install -r requirements.txt

4. Jalankan aplikasi:
   streamlit run app.py

5. Akses aplikasi melalui browser:
   http://localhost:8501

## Catatan
Proyek ini dikembangkan untuk keperluan akademik dan pembelajaran Machine Learning.
Hasil prediksi yang dihasilkan tidak dimaksudkan sebagai diagnosis atau keputusan
final, melainkan sebagai simulasi penerapan metode klasifikasi berbasis data.
