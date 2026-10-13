# Praktikum Pengolahan Citra Digital - Pertemuan 3
## Aplikasi Resize Citra & Visualisasi Matriks Piksel

Aplikasi interaktif berbasis **Python** dan **Streamlit** untuk mempelajari representasi spasial citra digital dalam bentuk matriks numerik (piksel) serta efek transformasi ukuran (resize) dengan berbagai metode interpolasi spasial.

---

### 🌟 Fitur Utama

1. **Input Citra Fleksibel**:
   - Upload citra digital sendiri (format PNG, JPG, JPEG, BMP, WEBP).
   - Disediakan citra sampel edukasi bawaan:
     - `8x8 Pola Matriks Edukasi`: Dirancang khusus agar setiap angka piksel dapat diamati secara utuh tanpa terpotong.
     - `16x16 Gradien Warna RGB`: Memperlihatkan variasi gradasi matriks channel warna.
     - `128x128 Pola Geometris Lengkap`: Untuk mengamati kualitas visual hasil interpolasi.

2. **Transformasi Ukuran (Resize)**:
   - **Metode Skala**: Persentase pembesaran/pengecilan (10% hingga 400%).
   - **Metode Dimensi Eksplisit**: Tentukan ukuran lebar ($W$) dan tinggi ($H$) baru dalam satuan piksel.
   - Opsi *Maintain Aspect Ratio* (pertahankan rasio aspek agar citra tidak terdistorsi).

3. **5 Algoritma Interpolasi Spasial**:
   - `Nearest Neighbor (cv2.INTER_NEAREST)`: Duplikasi piksel terdekat (cepat, pixelated).
   - `Bilinear (cv2.INTER_LINEAR)`: Rata-rata tertimbang 4 piksel tetangga ($2 \times 2$).
   - `Bicubic (cv2.INTER_CUBIC)`: Interpolasi polinomial kubik 16 piksel tetangga ($4 \times 4$).
   - `Area Resampling (cv2.INTER_AREA)`: Resampling berbasis relasi luas piksel (optimal untuk downscaling/pengecilan).
   - `Lanczos (cv2.INTER_LANCZOS4)`: Interpolasi fungsi sinc 64 tetangga ($8 \times 8$).

4. **🎯 Fitur Baru: Inspeksi Interaktif Dua Arah (Citra ⇄ Matriks)**:
   - **Klik pada Citra**: Klik sembarang titik pada citra langsung di layar untuk membaca nilai piksel (RGB, Grayscale, Hex) serta menampilkan jendela matriks tetangga lokal ($3 \times 3$ hingga $9 \times 9$).
   - **Pilih Angka di Matriks**: Memilih koordinat baris ($Y$) dan kolom ($X$) akan langsung menandai titik tersebut dengan garis bidik (*crosshair*) merah dan menampilkan pembesaran mikroskopis $25\times$ (zoom patch bergaris grid piksel).

5. **✨ Fitur Baru: Operasi Titik (Point Operations)**:
   - **Kecerahan (Brightness)**: $g(x, y) = \text{clip}(f(x, y) + \beta, 0, 255)$ dengan komparasi visual, pergeseran histogram, dan matriks sebelum/sesudah.
   - **Kontras (Contrast)**: Pilihan metode linier atau berpusat pada midpoint 128 $g(x, y) = \text{clip}(\alpha \cdot (f(x, y) - 128) + 128, 0, 255)$.
   - **Kombinasi Linier (α & β)**: Pengaturan simultan kontras dan kecerahan $g(x, y) = \text{clip}(\alpha \cdot f(x, y) + \beta, 0, 255)$.
   - **Ekualisasi Histogram (Histogram Equalization)**: Meratakan distribusi probabilitas kumulatif (CDF) tingkat keabuan dengan mode Global Equalization dan Adaptive CLAHE, lengkap dengan kurva CDF dan grafik histogram komparatif.

6. **Visualisasi Matriks Piksel yang Komprehensif**:
   - **Matriks Grayscale**: Representasi matriks 2D ($H \times W$) dengan nilai intensitas $[0 - 255]$.
   - **Matriks Channel RGB Terpisah**: Saluran Merah (Red), Hijau (Green), dan Biru (Blue).
   - **Tiga Mode Tampilan Matriks**:
     1. *Tabel Interaktif (DataFrame)* dengan pewarnaan gradien sesuai intensitas piksel (dioptimasi aman untuk citra berdimensi besar).
     2. *Peta Panas (Heatmap Grafis)* dengan anotasi angka piksel langsung di atas sel.
     3. *NumPy Raw Array* untuk melihat sintaks array pemrograman Python.
   - **Fitur ROI / Sub-Matriks Slicer**: Memilih rentang baris dan kolom yang ingin diperiksa secara mendetail pada citra berukuran besar.

7. **Analisis Statistik & Perbandingan**:
   - Ringkasan statistik piksel (Min, Max, Rata-rata, Standar Deviasi) sebelum dan sesudah resize.
   - Histogram distribusi intensitas piksel (Red, Green, Blue).

8. **Ekspor & Unduh**:
   - Unduh citra hasil resize (format PNG).
   - Unduh citra Grayscale hasil resize (format PNG).
   - Unduh matriks piksel ke dalam file **CSV** (Grayscale, Red, Green, Blue) untuk analisis di Excel atau software pengolah data lainnya.

---

### 🚀 Cara Menjalankan Aplikasi

1. Buka terminal pada folder direktori ini:
   ```bash
   cd "/Users/mac/Documents/Materi Pengajaran-2025/15.Pengolahan Citra Digital/Praktikum/Pertemuan 3"
   ```

2. Jalankan aplikasi Streamlit:
   ```bash
   streamlit run app.py
   ```

3. Browser akan otomatis terbuka pada alamat:
   ```
   http://localhost:8501
   ```

---

### 📂 Struktur Direktori

```
Pertemuan 3/
├── app.py                      # Skrip utama aplikasi Streamlit
├── generate_samples.py         # Skrip generator citra sampel edukasi
├── requirements.txt            # Daftar dependensi library
├── README.md                   # Dokumentasi panduan praktikum
└── sample_images/              # Folder citra sampel edukatif
    ├── contoh_8x8_pola_matriks.png
    ├── contoh_16x16_gradien_warna.png
    └── contoh_128x128_geometris.png
```

---

### 📖 Konsep Dasar Pengolahan Citra Digital

1. **Citra Digital sebagai Matriks**:
   Citra berdimensi $M$ baris dan $N$ kolom dengan saluran warna $C$ dimodelkan sebagai:
   $$f(y, x, c) \in [0, 255], \quad \text{tipe data: } \text{uint8}$$
   - $y$: Indeks baris (koordinat vertikal, $0 \le y < M$)
   - $x$: Indeks kolom (koordinat horizontal, $0 \le x < N$)
   - $c$: Indeks channel ($0 = \text{Red}, 1 = \text{Green}, 2 = \text{Blue}$)

2. **Konversi RGB ke Grayscale**:
   Mengikuti standar ITU-R BT.601:
   $$Y = 0.299 \times R + 0.587 \times G + 0.114 \times B$$
