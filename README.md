## Domain Proyek

Penyakit jantung atau yang dikenal dengan penyakit kardiovaskular adalah salah satu penyakit yang menyebabkan kematian secara global, diperkirakan kematian dari penyakit ini mencapai 17,9 juta dalam tahun 2019. Sangat penting untuk mendeteksi penyakit seawal mungkin agar bisa dicegah seperti konseling beserta pemberian obat-obat bisa dimulai

## Business Understanding

### Problem Statements

Berdasarkan kondisi yang telah diuraikan sebelumnya, pihak rumah sakit akan mengembangkan sebuah sistem prediksi kehadiran dari penyakit jantung untuk menjawab permasalahan berikut.

1. Dapatkah model machine learning memprediksi risiko penyakit jantung secara akurat menggunakan fitur-fitur yang tersedia dalam dataset?
2. Di antara algoritma K-Nearest Neighbors (KNN) dan Random Forest Classifier, manakah yang lebih efektif dalam memprediksi risiko penyakit jantung berdasarkan performa klasifikasi?

### Goals

Untuk menjawab pertanyaan tersebut, pihak rumah sakit akan membuat predictive modelling dengan tujuan atau goals sebagai berikut:

1. Mengetahui kemampuan model machine learning dalam memprediksi risiko penyakit jantung berdasarkan fitur-fitur yang ada.
2. Membandingkan performa algoritma K-Nearest Neighbors (KNN) dan Random Forest Classifier untuk menentukan mana yang lebih optimal dalam prediksi penyakit jantung.

## Data Understanding

Data yang digunakan Heart Disease Classification Dataset

link : https://www.kaggle.com/datasets/bharath011/heart-disease-classification-dataset

Dataset ini memiliki 1319 baris data dengan 9 kolom

1. Pengecekan Missing Values:
   dengan menggunakan **df.isna().sum()**, saya bisa menampilkan jumlah data dari kolom apa saja yang missing. Pada kasus ini, tidak ada data yang missing

2. Pengecekan Duplicated Values:
   dengan menggunakan **df.duplicated().sum()**, saya bisa mengetahui apakah ada data yang duplikat. Untuk dataset ini, tidak terdapat duplikat.

3. Pengecekan Outliers:
   untuk mengetahui apakah ada outliers atau tidak, saya menggunakan fungsi dari seaborn yaitu **sns.boxplot()**, visualisasi ini dapat mempermudah saya untuk memahami apakah data tersebut mengandung outliers

   ![Gambar Boxplot](https://builtin.com/sites/www.builtin.com/files/styles/ckeditor_optimize/public/inline-images/1_boxplots.jpg "Gambar Boxplot")

   Untuk menghilangkan Outliers, saya menggunakan IQR (Interquantile Range). Dengan IQR saya mendapatkan kalau dalam dataset ini terdapat outliers sebanyak **530** sampel. Kemudian saya menghapus outliers-nya dan tersisa **789** sampel

Setelah semua hal tersebut, dataset saya sudah bersih dan siap untuk dipakai.

### Variabel-variabel pada Heart Diseases Dataset adalah sebagai berikut:

1. age : Umur dari pasien
2. gender : Jenis kelamin dari pasien
3. impulse : Sinyal listrik yang dihasilkan oleh jantung
4. preassurehigh : Tekanan maksimum saat jantung berkontraksi
5. preassurelow : Tekanan minimum saat jantung berkontraksi
6. glucose : Kadar gula darah
7. kcm : Tiga mineral elektrolit yang sangat penting bagi fungsi jantung (Kalium(k), Kalsium(c), dan Magnesium(m))
8. troponin : Protein yang dilepaskan ke darah saat otot jantung rusak
9. class : Target dari model

## Data Preparation

Persiapan data yang dilakukan:

1. Pembersihan Outliers
   -> _Outlier_ adalah data ekstrem yang dapat secara signifikan mendistorsi parameter model machine learning, menurunkan akurasinya, dan memengaruhi metrik evaluasi. Karena beberapa model sangat sensitif terhadapnya, penanganan outlier penting untuk membangun model yang lebih andal dan akurat. Seperti yang dijelaskan dalam data Understanding, saya menggunakan boxplot untuk mengetahui apakah ada _outlier_ dalam dataset saya dan kemudian membersihkannya dengan menggunakan IQR (Interquantile Range). outliers dalam dataset inni mencapai **530** sampel.

2. Label Encode
   -> Label Encoding dilakukan saat analisis multivariate pada bagian pairplot dan heatmap untuk analisis lebih lanjut, label encoding ini bertujuan untuk mengubah data kategorikal menjadi data numerik, pada algoritma K-Nearest Neighbors dan RandomForestClassifier juga membutuhkan target yang berupa numerik(sudah di-_encode_).

3. Data Splitting
   -> Dilakukan untuk membagi data latih dan data uji dengan perbandingan 80:20, 80% data latih dan 20% data uji. Data splitting dilakukan untuk menghindari _overfitting_ pada model beserta data uji digunakan sebagai evaluasi model.

4. Data Scaling
   -> Dilakukan untuk mengubah nilai-nilai dari fitur agar memiliki skali nilai yang sama, serta mengatasi masalah skala yang berbeda-beda setiap fitur. Tanpa data scaling, algoritma machine learning dapat mengalami kesulitan dalam menemukan solusi yang optimal karena fitur-fitur yang memiliki skala yang berbeda akan memberikan bobot yang berbeda pada model.

## Modelling

Algoritma yang digunakan adalah:

1. K-Nearest Neighbors dengan parameter _neighbors=10_ (parameter lainnya adalah default atau bawaan model)

- KNN adalah algoritma klasifikasi yang sederhana. Untuk menentukan kelas dari sebuah data baru, KNN melihat 'k' tetangga terdekatnya dalam ruang fitur (data training yang sudah ada).

- Kelebihan KNN: Sederhana untuk dipahami dan diimplementasikan, efektif untuk dataset di mana batas keputusan tidak linear.
- Kekurangan KNN: Bisa menjadi lambat secara komputasional pada dataset yang sangat besar karena perlu menghitung jarak ke semua titik data training untuk setiap prediksi baru. Sensitif terhadap penskalaan fitur (fitur dengan rentang nilai besar bisa mendominasi perhitungan jarak).

2. Random Forest Classifier dengan parameter _random_state=42_ (parameter lainnya adalah default atau bawaan model)

RandomForestClassifier adalah algoritma ensemble learning yang membangun banyak pohon keputusan (decision tree) secara acak selama proses training.

- Kelebihan Random Forest: Sangat efektif dan sering memberikan akurasi yang tinggi, robust terhadap outlier dan noise, menangani data berdimensi tinggi dengan baik, mengurangi overfitting dibandingkan satu pohon keputusan, dan dapat memberikan estimasi pentingnya fitur.
- Kekurangan Random Forest: Bisa menjadi black box (sulit diinterpretasikan secara detail bagaimana keputusan dibuat dibandingkan satu pohon keputusan), membutuhkan lebih banyak sumber daya komputasi (waktu dan memori) dibandingkan algoritma yang lebih sederhana, terutama dengan jumlah pohon yang besar.

## Evaluation

Metrik yang saya gunakan untuk model klasifikasi ini adalah Accuracy, Precision, Recall, dan F1-Score

- **Accuracy** adalah Persentase prediksi benar (True Positive + True Negative) dibandingkan dengan seluruh prediksi.

- **Precision** adalah Dari semua prediksi positif, berapa banyak yang benar-benar positif.

- **Recall** adalah Dari semua kasus positif aktual, berapa banyak yang berhasil diprediksi.

- **F1-Score** adalah Rata-rata harmonik dari precision dan recall. Berguna ketika perlu menyeimbangkan keduanya.

* Tabel Evaluasi

| Metriks   | K-Nearest Neighbors | RandomForestClassifier |
| --------- | ------------------- | ---------------------- |
| Accuracy  | 0.75                | 0.97                   |
| Precision | 0.72                | 0.98                   |
| Recall    | 0.95                | 0.95                   |
| F1-Score  | 0.57                | 0.96                   |

- Confusion Matrix

  ![Confusion Matrix KNN](https://i.imgur.com/wn0PIl8.png "Confusion Matrix KNN")
  ![Confusion Matrix RFC](https://i.imgur.com/WJKrLdV.png "Confusion Matrix RFC")

- Penjelsan:

1. **True-True (True Positive):**

   - Pengertian : Model dengan benar memprediksi bahwa suatu data termasuk dalam kelas positif.
   - Model K-Nearest Neighbors : 92 Sampel
   - Model RandomForestClassifier : 101 Sampel

2. **True-False (False Negative):**

   - Pengertian : Model salah memprediksi bahwa suatu data termasuk dalam kelas Negatif, padahal sebenarnya data tersebut adalah Positif. Ini berarti ada kasus positif yang terlewatkan.
   - Model K-Nearest Neighbors : 10 Sampel
   - Model RandomForestClassifier : 1 Sampel

3. **False-True (False Positive):**

   - Pengertian : Model Anda salah memprediksi bahwa suatu data termasuk dalam kelas Positif, padahal sebenarnya data tersebut adalah Negatif.
   - Model K-Nearest Neighbors : 30 Sampel
   - Model RandomForestClassifier : 3 Sampel

4. **False-False (True Negative):**
   - Pengertian : Model dengan benar memprediksi bahwa suatu data termasuk dalam kelas Negatif.
   - Model K-Nearest Neighbors : 26 Sampel
   - Model RandomForestClassifier : 53 Sampel

- **Pertanyaan 1:**
  ternyata algoritma machine learning dapat memprediksi risiko penyakit jantung dengan fitur-fitur yang ada dengan hampir sempurna jika dilihat dari metriks algoritma Random Forest Classifier.

- **Pertanyaan 2:**
  Dari akurasi tersebut dapat saya simpulkan kalau model RandomForestClassifer adalah model yang tepat untuk mesalah berikut dikarenakan metriks yang dihasilkan hampir sempurna.
