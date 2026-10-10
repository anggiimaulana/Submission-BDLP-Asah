# Klasifikasi Gambar Histopatologi Kanker Paru dengan CNN

Proyek ini membangun model **Convolutional Neural Network (CNN)** untuk mengklasifikasikan citra histopatologi ke dalam tiga kelas: `adenocarcinoma`, `benign`, dan `squamous_cell_carcinoma`. Model dikembangkan menggunakan TensorFlow/Keras dan disiapkan dalam format SavedModel, TensorFlow Lite (TFLite), serta TensorFlow.js (TFJS).

> **Penting:** Proyek ini dibuat untuk tujuan pembelajaran dan eksperimen machine learning, bukan untuk diagnosis, skrining, atau pengambilan keputusan medis.

## Dataset

- **Sumber:** [Lung Cancer Histopathological Images — Kaggle](https://www.kaggle.com/datasets/rm1000/lung-cancer-histopathological-images)
- **Jumlah gambar:** 15.000 gambar.
- **Kelas:**
  - `adenocarcinoma`
  - `benign`
  - `squamous_cell_carcinoma`
- **Resolusi gambar sumber:** 768 × 768 piksel.
- **Lisensi yang tercantum saat dataset diunduh:** CC BY-SA 4.0. Periksa kembali halaman dataset untuk ketentuan atribusi dan penggunaan terkini.

## Pembagian Dataset

Dataset dibagi secara stratified agar proporsi kelas tetap terjaga.

| Subset | Jumlah gambar | Proporsi |
|---|---:|---:|
| Train | 9.600 | 64% |
| Validation | 2.400 | 16% |
| Test | 3.000 | 20% |
| **Total** | **15.000** | **100%** |

## Preprocessing dan Augmentasi

- Gambar diubah ke mode grayscale dan diubah ukurannya menjadi 150 × 150 piksel.
- Nilai piksel dinormalisasi dengan skala `1/255`.
- Augmentasi pada data training meliputi rotasi, pergeseran lebar/tinggi, horizontal flip, dan zoom.
- Data validation dan test hanya dinormalisasi tanpa augmentasi.
- Batch size training dan validation adalah 32.

## Arsitektur Model

Model dibangun menggunakan `Sequential` pada Keras dengan komponen utama berikut:

1. Empat blok konvolusi menggunakan `Conv2D` dengan 32, 64, 128, dan 256 filter.
2. Setiap blok dilengkapi `BatchNormalization` dan `MaxPooling2D`.
3. Lapisan klasifikasi menggunakan `Flatten`, `Dense(512, activation="relu")`, dan `Dropout(0.5)`.
4. Lapisan output menggunakan `Dense(3, activation="softmax")` untuk menghasilkan prediksi tiga kelas.

Konfigurasi training menggunakan optimizer Adam, loss `categorical_crossentropy`, metrik accuracy, serta callback `EarlyStopping`, `ModelCheckpoint`, dan `ReduceLROnPlateau`. Jumlah epoch maksimum yang ditetapkan adalah 30.

## Hasil Evaluasi

Berdasarkan output evaluasi yang tersimpan pada notebook:

| Metrik | Hasil |
|---|---:|
| Akurasi training tertinggi yang tercatat | 97,07% |
| Akurasi pada test set | 96,50% |

Notebook juga menampilkan plot accuracy dan loss untuk training/validation, confusion matrix, classification report, analisis kesalahan klasifikasi, serta visualisasi Grad-CAM.

Hasil ini merupakan evaluasi eksperimen pada dataset yang digunakan dan bukan bukti validasi untuk penggunaan klinis.

## Format Model

Model disiapkan dalam tiga format:

- **SavedModel:** untuk penggunaan melalui ekosistem TensorFlow.
- **TFLite:** untuk perangkat yang mendukung TensorFlow Lite. Folder ini berisi `model.tflite` dan `label.txt`.
- **TensorFlow.js:** untuk aplikasi JavaScript/browser. Folder ini berisi `model.json` dan seluruh file bobot `.bin` yang dirujuk oleh manifest.

Struktur berkas submission:

```text
submission/
├── tfjs_model/
│   ├── model.json
│   └── group1-shard*.bin
├── tflite/
│   ├── model.tflite
│   └── label.txt
├── saved_model/
│   ├── saved_model.pb
│   └── variables/
├── notebook.ipynb
├── README.md
└── requirements.txt
```

Pastikan semua file `.bin` hasil ekspor TFJS tetap berada bersama `model.json`. Jangan mengganti nama shard secara manual karena nama file tersebut dirujuk oleh manifest.

## Cara Menjalankan Notebook

1. Buka `notebook.ipynb` menggunakan Google Colab atau lingkungan Python yang kompatibel.
2. Siapkan kredensial Kaggle melalui Colab Secrets dengan nama `KAGGLE_USERNAME` dan `KAGGLE_KEY`, atau melalui environment variable yang sesuai. Jangan menuliskan API key langsung di notebook atau README.
3. Jalankan sel notebook secara berurutan untuk mengunduh dataset, menyiapkan data, melatih dan mengevaluasi model, lalu mengekspor model.
4. Pastikan file keluaran tersedia dan struktur direktori mengikuti struktur submission di atas.

Lingkungan yang tercatat ketika notebook dijalankan adalah Google Colab dengan Python 3.13.15 dan TensorFlow 2.21.0. Versi dependency dapat bergantung pada runtime yang digunakan. Sertakan `requirements.txt` yang mencatat pustaka utama dan versi yang relevan untuk proyek ini; tidak semua paket bawaan Google Colab perlu dimasukkan.

## Identitas Proyek

- **Nama:** Anggi Maulana
- **ID Dicoding:** `anggi_maulana_07`