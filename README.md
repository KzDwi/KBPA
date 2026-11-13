
# Klasifikasi Spesies Ikan

Aplikasi web berbasis Streamlit untuk mengklasifikasi spesies ikan menggunakan model Custom Convolutional Neural Network (CNN) yang dilatih dari awal (from scratch).

Akses aplikasi: **https://klasifihkasi.streamlit.app/**

---

## Daftar Isi

- [Tentang Aplikasi](#tentang-aplikasi)
- [Fitur](#fitur)
- [Cara Menggunakan Aplikasi Online](#cara-menggunakan-aplikasi-online)
- [Cara Menjalankan Secara Lokal](#cara-menjalankan-secara-lokal)
- [Struktur File](#struktur-file)
- [Persyaratan Sistem](#persyaratan-sistem)
- [Detail Teknis](#detail-teknis)
- [Pemecahan Masalah](#pemecahan-masalah)
- [Batasan dan Catatan Penting](#batasan-dan-catatan-penting)

---

## Tentang Aplikasi

Aplikasi ini menerima input berupa gambar ikan, baik melalui unggahan file maupun melalui tautan URL langsung ke file gambar. Model CNN akan memproses gambar tersebut dan mengembalikan:

- Nama spesies ikan hasil prediksi
- Tingkat keyakinan (confidence) dalam persen
- Probabilitas untuk setiap kelas yang dikenali model

Model dilatih menggunakan dataset citra ikan dan disimpan dalam format `.keras`. Konfigurasi preprocessing (ukuran gambar dan faktor rescale) disimpan terpisah agar konsisten antara proses training dan inference.

---

## Fitur

- Unggah gambar langsung dari perangkat (format JPG, JPEG, PNG)
- Ambil gambar dari URL publik yang mengarah langsung ke file gambar
- Menampilkan hasil prediksi berupa label kelas dan nilai confidence
- Menampilkan probabilitas untuk seluruh kelas dalam bentuk bar progres
- Validasi URL: skema, Content-Type, dan batas ukuran file (maksimal 15 MB)
- Cache model dan konfigurasi agar aplikasi tidak memuat ulang setiap interaksi
- Antarmuka dua tab: "Unggah File" dan "Dari Link / URL"

---

## Cara Menggunakan Aplikasi Online

Aplikasi dapat langsung digunakan tanpa instalasi melalui tautan berikut:

**https://klasifihkasi.streamlit.app/**

### Langkah Penggunaan

1. Buka tautan aplikasi di browser (Chrome, Firefox, Edge, atau Safari versi terbaru).
2. Tunggu hingga proses "Memuat model..." selesai. Proses ini hanya terjadi sekali per sesi.
3. Pilih salah satu dari dua tab yang tersedia.

### Tab 1: Unggah File

1. Klik tab **Unggah File**.
2. Klik tombol **Browse files** atau area unggah.
3. Pilih file gambar ikan dari perangkat Anda dengan format JPG, JPEG, atau PNG.
4. Aplikasi akan otomatis menampilkan gambar, hasil prediksi, nilai confidence, dan probabilitas per kelas.

### Tab 2: Dari Link / URL

1. Klik tab **Dari Link / URL**.
2. Tempelkan URL langsung ke file gambar pada kolom yang tersedia. Contoh yang benar:
   - `https://example.com/ikan.jpg`
   - `https://cdn.contoh.com/fish/ikan-koi.png`
3. Klik tombol **Ambil & Prediksi**.
4. Aplikasi akan mengunduh gambar, menampilkannya, lalu menampilkan hasil prediksi.

Catatan penting untuk tab URL:

- URL harus diawali dengan `http://` atau `https://`.
- URL harus mengarah langsung ke file gambar, bukan ke halaman web yang menampilkan gambar. Contoh URL yang salah: halaman hasil pencarian Google Images, halaman artikel, atau halaman galeri.
- Server sumber gambar harus mengembalikan header `Content-Type` yang diawali dengan `image/`.
- Ukuran file gambar tidak boleh melebihi 15 MB.
- Waktu unduh dibatasi 10 detik. Jika server lambat, proses akan gagal.

### Membaca Hasil Prediksi

Setelah gambar diproses, aplikasi menampilkan:

- **Gambar input** di kolom kiri beserta keterangan sumbernya.
- **Hasil Prediksi** di kolom kanan berupa nama kelas dengan highlight hijau.
- **Confidence** dalam bentuk metrik persentase.
- **Probabilitas per Kelas** berupa daftar bar yang diurutkan dari probabilitas tertinggi ke terendah. Setiap bar menampilkan nama kelas dan persentase.

---

## Cara Menjalankan Secara Lokal

Jika Anda ingin menjalankan aplikasi di komputer sendiri, ikuti langkah berikut.

### 1. Prasyarat

- Python versi 3.9 hingga 3.11 (disarankan 3.10 atau 3.11)
- pip versi terbaru
- Koneksi internet untuk mengunduh dependensi

### 2. Unduh Proyek

Letakkan file berikut dalam satu direktori yang sama:


### 2. Unduh Proyek

Letakkan file berikut dalam satu direktori yang sama:

```
project-folder/
├── app.py
├── requirements.txt
├── fish_cnn_model.keras
├── class_indices.json
└── preprocess_config.json
```

### 3. Buat Virtual Environment (Disarankan)

Linux atau macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows (Command Prompt):

```bash
python -m venv venv
venv\Scripts\activate
```

Windows (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 4. Instal Dependensi

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Isi `requirements.txt`:

```
tensorflow==2.16.1
streamlit==1.35.0
pillow==10.3.0
numpy==1.26.4
```

Catatan: `requests` sudah menjadi dependensi Streamlit, namun jika terjadi error `ModuleNotFoundError: No module named 'requests'`, jalankan:

```bash
pip install requests
```

### 5. Jalankan Aplikasi

```bash
streamlit run app.py
```

Setelah perintah dijalankan, Streamlit akan membuka browser secara otomatis pada alamat default:

```
http://localhost:8501
```

Jika browser tidak terbuka otomatis, salin alamat tersebut ke browser secara manual.

### 6. Menghentikan Aplikasi

Tekan `Ctrl + C` pada terminal tempat aplikasi berjalan.

---

## Struktur File

| File | Deskripsi |
|------|-----------|
| `app.py` | Berkas utama aplikasi Streamlit yang berisi logika UI, pemuatan model, pengambilan gambar dari URL, preprocessing, dan prediksi. |
| `requirements.txt` | Daftar dependensi Python beserta versi yang dipakai. |
| `fish_cnn_model.keras` | Model CNN yang telah dilatih, disimpan dalam format Keras native. |
| `class_indices.json` | Pemetaan nama kelas ke indeks numerik, contoh: `{"ikan_koi": 0, "ikan_lele": 1, ...}`. |
| `preprocess_config.json` | Konfigurasi preprocessing berisi `image_size` (misal `[224, 224]`) dan `rescale_factor` (misal `0.00392156862745098`). |

Ketiga file model dan konfigurasi (`fish_cnn_model.keras`, `class_indices.json`, `preprocess_config.json`) harus berada di direktori yang sama dengan `app.py`. Jika tidak, aplikasi akan gagal saat startup.

---

## Persyaratan Sistem

### Untuk Penggunaan Online

- Browser modern: Chrome 90+, Firefox 88+, Edge 90+, atau Safari 14+
- Koneksi internet stabil
- JavaScript diaktifkan

### Untuk Menjalankan Lokal

- Sistem operasi: Windows 10/11, Linux (Ubuntu 20.04+), atau macOS 11+
- RAM minimal 4 GB (disarankan 8 GB karena TensorFlow memuat model ke memori)
- Ruang disk minimal 2 GB untuk dependensi dan model
- CPU dengan dukungan AVX (umumnya tersedia pada prosesor keluaran 2011 ke atas)

---

## Detail Teknis

### Arsitektur Model

Model merupakan Custom CNN yang dilatih dari awal, bukan model pre-trained. Model dimuat menggunakan `tf.keras.models.load_model` dari file `fish_cnn_model.keras`.

### Preprocessing Gambar

Preprocessing dilakukan agar identik dengan saat training:

1. Konversi gambar ke mode RGB.
2. Resize ke ukuran yang ditentukan `preprocess_config.json` (default 224x224) menggunakan interpolasi bilinear.
3. Konversi ke array `float32` dan kalikan dengan `rescale_factor` (default `1/255`).
4. Tambahkan dimensi batch sehingga bentuk menjadi `(1, H, W, 3)`.

### Prediksi

Model mengembalikan vektor probabilitas sepanjang jumlah kelas. Kelas dengan probabilitas tertinggi diambil sebagai label prediksi, dan nilai probabilitasnya menjadi confidence.

### Pengambilan Gambar dari URL

Fungsi `fetch_image_from_url` melakukan validasi berlapis:

- Skema URL harus `http` atau `https`.
- Header `User-Agent` disertakan agar tidak ditolak server.
- Status respons harus 200.
- `Content-Type` harus diawali `image/`.
- Ukuran konten dibatasi 15 MB dan diunduh secara streaming per 8192 byte.
- File harus dapat didekode oleh PIL; jika gagal, dianggap bukan gambar valid.

### Cache

- `st.cache_resource` dipakai untuk memuat model agar tidak dimuat ulang setiap interaksi.
- `st.cache_data` dipakai untuk memuat konfigurasi kelas dan preprocessing.
- Hasil fetch dari URL disimpan di `st.session_state` agar tidak mengunduh ulang saat pengguna berinteraksi dengan widget lain.

### Kompatibilitas Keras

Terdapat patch kecil pada `Dense.__init__` untuk mengabaikan argumen `quantization_config` yang kadang muncul pada model yang disimpan dengan versi Keras lebih baru. Patch ini memastikan model tetap dapat dimuat pada TensorFlow 2.16.1.

---

## Pemecahan Masalah

### Aplikasi menampilkan "Memuat model..." terlalu lama

Model berukuran cukup besar. Tunggu hingga selesai. Jika lebih dari dua menit, muat ulang halaman. Pada deployment Streamlit Cloud, kontainer mungkin sedang cold start.

### Error "Gagal mengunduh gambar"

Penyebab umum:

- URL bukan tautan langsung ke file gambar.
- Server memblokir permintaan dari lingkungan cloud.
- Koneksi timeout (lebih dari 10 detik).
- URL memerlukan autentikasi atau cookie.

Solusi: gunakan tautan langsung ke file gambar (berakhiran `.jpg`, `.png`, dan sejenisnya) dari server publik seperti Imgur, GitHub raw, atau CDN gambar.

### Error "URL bukan gambar (Content-Type: ...)"

Server mengembalikan halaman HTML, bukan file gambar. Pastikan URL mengarah ke file gambar, bukan halaman web.

### Error "Ukuran gambar melebihi batas 15 MB"

Kompres gambar terlebih dahulu atau gunakan versi gambar beresolusi lebih rendah.

### Error "File tidak dapat dibaca sebagai gambar"

File mungkin rusak atau formatnya tidak didukung. Coba buka file di aplikasi penampil gambar untuk memastikan file valid, lalu unggah ulang.

### Error saat menjalankan lokal: "No module named 'tensorflow'"

Virtual environment belum aktif atau dependensi belum terpasang. Aktifkan venv lalu jalankan `pip install -r requirements.txt`.

### Error versi TensorFlow tidak kompatibel

Pastikan Python yang dipakai adalah 3.9 hingga 3.11. TensorFlow 2.16.1 tidak mendukung Python 3.12 ke atas.

### Aplikasi lokal berjalan tetapi browser tidak terbuka

Buka `http://localhost:8501` secara manual. Jika port 8501 sedang dipakai, Streamlit akan memilih port lain; periksa output terminal untuk alamat yang benar.

---

## Batasan dan Catatan Penting

- Model hanya mengenali kelas yang ada pada dataset training. Gambar spesies di luar kelas tersebut tetap akan diprediksi ke salah satu kelas yang dikenal, kadang dengan confidence tinggi yang menyesatkan.
- Kualitas prediksi sangat bergantung pada kualitas gambar: pencahayaan, sudut pengambilan, dan latar belakang yang mirip dengan data training akan meningkatkan akurasi.
- Model merupakan CNN yang dilatih dari awal, sehingga akurasinya tidak setara dengan model pre-trained berskala besar. Gunakan hasil sebagai referensi, bukan keputusan final.
- Aplikasi tidak menyimpan gambar yang diunggah maupun URL yang dimasukkan ke basis data. Namun, perlu diketahui bahwa Streamlit Cloud dapat menyimpan log akses standar.
- Batas ukuran gambar 15 MB dan timeout 10 detik merupakan konfigurasi tetap pada versi ini.
- Aplikasi tidak memerlukan autentikasi dan dapat diakses siapa saja yang memiliki tautan.

---

## Lisensi dan Kredit

Aplikasi ini dibangun dengan:

- Streamlit
- TensorFlow dan Keras
- Pillow
- NumPy
- Requests

Model dilatih secara mandiri menggunakan dataset citra ikan.
