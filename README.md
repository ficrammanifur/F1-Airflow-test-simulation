# 🏎️ F1 Airflow Test Simulation

Simulasi aliran udara (*airflow*) untuk mobil Formula 1 menggunakan Python dan library visualisasi 3D.  
Project ini dibuat untuk membantu memvisualisasikan bagaimana udara mengalir di sekitar mobil F1 sebagai studi awal *aerodynamic analysis*.

## 📦 Struktur Project
```
F1-Airflow-test-simulation/
├── models/ # File model 3D (STL, OBJ, dll.)
├── scripts/ # Script Python untuk simulasi & visualisasi
├── data/ # Data pendukung (jika ada)
├── README.md
└── requirements.txt # Daftar dependensi Python
```

## ⚙️ Fitur
- Visualisasi model mobil F1 dalam 3D
- Simulasi aliran udara sederhana menggunakan Python
- Eksperimen mesh decimation & filter untuk mempercepat render
- (Coming soon) Perhitungan drag & downforce dasar

## 🚀 Cara Menjalankan
1. Clone repository:
   ```bash
   git clone https://github.com/ficrammanifur/F1-Airflow-test-simulation.git
   cd F1-Airflow-test-simulation
2.Install dependensi:
   ```bash
   pip install -r requirements.txt
```
3.Jalankan simulasi:
```bash
   g++ view_mesh.cpp o-main
   python view_mesh.py
```
---
## 🧰 Dependencies
pyvista
numpy
scipy

---
📌 Catatan
Project ini masih tahap eksperimen & belajar.
--
Model 3D disederhanakan (decimation) agar proses render lebih cepat.

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Python-3.x-yellow?logo=python" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>
</p>

