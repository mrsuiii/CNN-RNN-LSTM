# Analisis Hyperparameter Recurrent Neural Network (RNN) untuk Klasifikasi Teks

## Deskripsi Singkat Repository

Repository ini berisi implementasi dan analisis pengaruh hyperparameter pada model Recurrent Neural Network (RNN), khususnya SimpleRNN dan Long Short-Term Memory (LSTM), untuk tugas klasifikasi teks. Analisis dilakukan dengan memvariasikan jumlah layer, jumlah cell per layer, dan jenis layer (unidirectional/bidirectional). Metrik utama yang digunakan untuk perbandingan adalah Macro F1-score, serta observasi terhadap training loss dan validation loss. Dataset yang digunakan adalah data teks yang dibagi menjadi set pelatihan, validasi, dan pengujian.

## Cara Setup dan Run Program

Proyek ini sangat disarankan untuk dijalankan di lingkungan Google Colaboratory (Google Colab) karena ketergantungan pada akses Google Drive untuk dataset dan sumber daya komputasi (GPU/TPU) yang tersedia.

### 1. Prasyarat

Pastikan Anda memiliki akun Google dan akses ke Google Colab.

### 2. Struktur Dataset

Anda harus menempatkan file dataset berikut di dalam folder **`dataset_rnn`** (atau nama folder lain yang Anda sesuaikan) di Google Drive Anda:
* `train.csv`
* `valid.csv`
* `test.csv`

**Contoh Path di Google Drive Anda:**
`My Drive/dataset_rnn/train.csv`
`My Drive/dataset_rnn/valid.csv`
`My Drive/dataset_rnn/test.csv`

### 3. Menjalankan Kode di Google Colab

1.  Buka Google Colab (https://colab.research.google.com/).
2.  Buat Notebook baru.
3.  Salin seluruh kode Python yang telah disediakan (misalnya, yang terakhir kali saya berikan untuk eksperimen LSTM) ke dalam sel kode di Notebook Colab Anda.
4.  **Mount Google Drive Anda:** Jalankan sel pertama yang berisi kode untuk me-mount Google Drive:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
    Ikuti instruksi otorisasi yang muncul.
5.  **Sesuaikan Path Data:** Pastikan variabel `BASE_DRIVE_PATH` dalam kode Anda telah disesuaikan dengan benar ke lokasi folder dataset Anda di Google Drive:
    ```python
    BASE_DRIVE_PATH = '/content/drive/MyDrive/dataset_rnn/' # Sesuaikan ini
    ```
6.  **Sesuaikan Kolom Target:** Ganti `'target'` dengan nama kolom label yang benar di dataset Anda (berdasarkan percakapan kita, ini adalah `'label'`):
    ```python
    TARGET_COLUMN = 'label' # Pastikan ini sesuai dengan nama kolom label Anda
    ```
7.  **Jalankan Semua Sel:** Jalankan semua sel kode secara berurutan. Program akan memuat data, melakukan preprocessing, membangun dan melatih model dengan berbagai konfigurasi hyperparameter, serta menampilkan hasil dan grafik.

### 4. Dependensi (Libraries)

Kode ini memerlukan library Python berikut, yang sebagian besar sudah tersedia di lingkungan Google Colab secara default:
* `pandas`
* `numpy`
* `matplotlib`
* `scikit-learn` (untuk `LabelEncoder` dan `f1_score`)
* `tensorflow` / `keras`

Jika ada library yang belum terinstal, Anda bisa menginstalnya di sel Colab dengan:
```bash
!pip install nama_library
```

| NIM         | Nama Anggota                    | Peran/Tugas         |
| :---------- | :-------------------------------| :------------------ |
| 13522079    | Emery Fathan Zwageri            | RNN, LSTM           |
| 13522089    | Abdul Rafi Radityo Hutomo       | CNN                 |
| 13522097    | Ellijah Darrelshane Suryanegara | RNN, LSTM           |
