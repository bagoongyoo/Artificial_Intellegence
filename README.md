# DCGAN Implementation for Anime Face Generation

Repository ini berisi implementasi **Deep Convolutional Generative Adversarial Networks (DCGAN)** menggunakan PyTorch untuk men-generate wajah karakter anime. Proyek ini dibuat sebagai Final Project mata kuliah Kecerdasan Buatan.

Model dilatih menggunakan dataset wajah anime berukuran 64x64 pixel untuk menghasilkan gambar sintetik baru yang belum pernah ada sebelumnya.

## 📋 Deskripsi Project

Tujuan utama proyek ini adalah membangun model generatif yang stabil menggunakan arsitektur DCGAN standar (berdasarkan paper Radford et al.). 

**Fitur Utama:**
* **Generator:** Menggunakan *Transposed Convolution* untuk upsampling dari noise vector (100) menjadi gambar RGB (64x64).
* **Discriminator:** Menggunakan CNN standar untuk klasifikasi biner (Real vs Fake).
* **Optimasi:** Menggunakan Adam Optimizer dengan learning rate `0.0002` dan beta1 `0.5`.
* **Stabilitas:** Menerapkan *Label Smoothing* (target 0.9) untuk mencegah *Discriminator Overpowering* dan menjaga kestabilan grafik Loss.

## 📂 Dataset

Dataset yang digunakan adalah kumpulan wajah anime yang telah di-crop.
* **Sumber:** `anime_faces.zip` https://www.kaggle.com/datasets/soumikrakshit/anime-faces
* **Resolusi Input:** 64x64 pixel
* **Format:** RGB
* **Preprocessing:** Normalisasi ke range [-1, 1] sesuai standar Tanh pada output Generator.

## 🛠️ Requirements & Instalasi

Project ini dikembangkan menggunakan Python 3.10 di lingkungan Windows dengan dukungan GPU (CUDA).

### Prasyarat
* Python 3.8+
* CUDA Toolkit 12.1 (Untuk training via GPU)
* Visual Studio Code (Recommended)

### Langkah Instalasi
1.  **Clone repository ini:**
    ```bash
    git clone [https://github.com/username-kamu/anime-dcgan.git](https://github.com/username-kamu/anime-dcgan.git)
    cd anime-dcgan
    ```

2.  **Buat Virtual Environment (Rekomendasi agar library tidak bentrok):**
    ```bash
    python -m venv .venv
    # Aktifkan environment:
    # Windows (PowerShell): .\.venv\Scripts\Activate.ps1
    # Windows (CMD): .venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    Disarankan menggunakan perintah berikut untuk memastikan PyTorch versi GPU terinstal dengan benar (hindari install CPU version):
    ```bash
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121) --no-cache-dir
    pip install numpy matplotlib jupyter ipykernel
    ```

## 🚀 Cara Menjalankan

1.  Pastikan file `anime_faces.zip` berada di root folder project.
2.  Buka file notebook utama (misal: `DCGAN_Training.ipynb`) di VS Code.
3.  Pastikan kernel yang dipilih adalah `.venv`.
4.  Jalankan cell secara berurutan.

**Catatan untuk Pengguna Windows:**
Pada konfigurasi DataLoader, parameter `num_workers` di-set ke `0` untuk menghindari `BrokenPipeError` atau `Multiprocessing Error` yang umum terjadi di Windows.
```python
workers = 0