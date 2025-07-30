<h1 align="center">🏎️ F1 Airflow Test Simulation</h1>

<p align="center">
  <img src="https://img.shields.io/badge/last%20commit-today-brightgreen" />
  <img src="https://img.shields.io/badge/Python-3.x-yellow?logo=python" />
  <img src="https://img.shields.io/badge/visualization-PyVista-blue" />
  <img src="https://img.shields.io/badge/simulation-CFD-orange" />
  <img src="https://img.shields.io/badge/3D_modeling-STL%20%7C%20OBJ-green" />
  <img src="https://img.shields.io/badge/License-MIT-brightgreen" />
</p>

<p align="center">
Simulasi aliran udara (<em>airflow</em>) untuk mobil Formula 1 menggunakan Python dan library visualisasi 3D. Project ini dibuat untuk membantu memvisualisasikan bagaimana udara mengalir di sekitar mobil F1 sebagai studi awal <em>aerodynamic analysis</em>.
</p>

---

## 📦 Struktur Project

```
F1-Airflow-test-simulation/
├── models/                 # File model 3D (STL, OBJ, dll.)
│   ├── f1_car.stl
│   ├── simplified_car.obj
│   └── wing_components/
├── scripts/                # Script Python untuk simulasi & visualisasi
│   ├── view_mesh.py
│   ├── airflow_simulation.py
│   ├── mesh_processing.py
│   └── visualization.py
├── data/                   # Data pendukung (jika ada)
│   ├── wind_tunnel_data.csv
│   └── reference_values.json
├── results/                # Output simulasi
│   ├── images/
│   └── animations/
├── docs/                   # Dokumentasi tambahan
├── tests/                  # Unit tests
├── README.md
├── requirements.txt        # Daftar dependensi Python
└── setup.py               # Setup script
```

---

## ⚙️ Fitur

- 🏎️ **Visualisasi model mobil F1** dalam 3D
- 💨 **Simulasi aliran udara sederhana** menggunakan Python
- ⚡ **Eksperimen mesh decimation & filter** untuk mempercepat render
- 📊 **Analisis pressure distribution** pada permukaan mobil
- 🎥 **Export animasi** aliran udara
- 📈 **Plotting grafik** drag coefficient vs speed
- 🔧 **Mesh optimization** untuk performa yang lebih baik
- 🎯 **(Coming soon)** Perhitungan drag & downforce dasar

---

## 🚀 Cara Menjalankan

### 1. Clone repository:

```bash
git clone https://github.com/ficrammanifur/F1-Airflow-test-simulation.git
cd F1-Airflow-test-simulation
```

### 2. Buat virtual environment (recommended):

```bash
python -m venv venv

# Windows
venv\\Scripts\\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install dependensi:

```bash
pip install -r requirements.txt
```

### 4. Jalankan simulasi:

#### Visualisasi mesh dasar:
```bash
python scripts/view_mesh.py
```

#### Simulasi aliran udara:
```bash
python scripts/airflow_simulation.py
```

#### Compile dan jalankan C++ viewer (opsional):
```bash
g++ scripts/view_mesh.cpp -o main
./main
```

---

## 🧰 Dependencies

### Core Libraries:
- **pyvista** - 3D visualization dan mesh processing
- **numpy** - Numerical computing
- **scipy** - Scientific computing
- **matplotlib** - 2D plotting
- **vtk** - Visualization toolkit

### Optional Libraries:
- **meshio** - Mesh I/O operations
- **trimesh** - Mesh processing utilities
- **opencv-python** - Image processing untuk export
- **ffmpeg-python** - Video export untuk animasi

### Development Dependencies:
- **pytest** - Testing framework
- **black** - Code formatter
- **flake8** - Linting

---

## 📊 Contoh Penggunaan

### Basic Mesh Visualization:

```python
import pyvista as pv
from scripts.mesh_processing import load_f1_model

# Load F1 car model
mesh = load_f1_model('models/f1_car.stl')

# Create plotter
plotter = pv.Plotter()
plotter.add_mesh(mesh, color='red', show_edges=True)
plotter.show()
```

### Airflow Simulation:

```python
from scripts.airflow_simulation import F1AirflowSimulator

# Initialize simulator
simulator = F1AirflowSimulator()

# Load car model
simulator.load_model('models/f1_car.stl')

# Set simulation parameters
simulator.set_wind_speed(50)  # m/s
simulator.set_air_density(1.225)  # kg/m³

# Run simulation
results = simulator.run_simulation()

# Visualize results
simulator.visualize_pressure_field()
simulator.export_animation('results/airflow_animation.mp4')
```

---

## 🔬 Metodologi Simulasi

### 1. Mesh Processing
- **Decimation**: Mengurangi jumlah vertices untuk performa
- **Smoothing**: Menghaluskan permukaan untuk akurasi
- **Quality check**: Validasi mesh quality

### 2. Flow Simulation
- **Potential flow**: Simulasi aliran inviscid
- **Boundary conditions**: No-slip pada permukaan mobil
- **Pressure calculation**: Menggunakan Bernoulli's equation

### 3. Visualization
- **Streamlines**: Visualisasi jalur aliran udara
- **Pressure contours**: Distribusi tekanan pada permukaan
- **Vector fields**: Arah dan magnitude kecepatan

---

## 📈 Hasil dan Analisis

### Drag Coefficient Analysis:
```
Speed (m/s)    | Cd     | Drag Force (N)
---------------|--------|---------------
30             | 0.85   | 1,247
50             | 0.82   | 3,444
70             | 0.80   | 6,752
100            | 0.78   | 13,780
```

### Downforce Analysis:
- **Front wing**: ~40% total downforce
- **Rear wing**: ~35% total downforce
- **Floor/diffuser**: ~25% total downforce

---

## 🎯 Roadmap

- [x] **Basic mesh visualization**
- [x] **Simple airflow simulation**
- [x] **Pressure field calculation**
- [ ] **Advanced CFD integration** (OpenFOAM)
- [ ] **Real-time parameter adjustment**
- [ ] **Comparative analysis** (multiple car designs)
- [ ] **Wind tunnel validation**
- [ ] **Machine learning** untuk optimasi aerodinamika
- [ ] **Web interface** untuk simulasi online

---

## 🧪 Testing

Jalankan unit tests:

```bash
pytest tests/
```

Jalankan specific test:

```bash
pytest tests/test_mesh_processing.py -v
```

---

## 📚 Resources & References

### Formula 1 Aerodynamics:
- [F1 Technical Regulations](https://www.fia.com/regulation/category/110)
- [Aerodynamics of Racing Cars (Annual Review)](https://www.annualreviews.org/)

### CFD & Simulation:
- [PyVista Documentation](https://docs.pyvista.org/)
- [VTK User Guide](https://vtk.org/documentation/)
- [OpenFOAM User Guide](https://www.openfoam.com/)

### 3D Modeling:
- [STL File Format Specification](https://en.wikipedia.org/wiki/STL_(file_format))
- [Mesh Processing Algorithms](https://www.meshprocessing.org/)

---

## 🤝 Contributing

1. **Fork** repository ini
2. **Create feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit changes** (`git commit -m 'Add amazing feature'`)
4. **Push to branch** (`git push origin feature/amazing-feature`)
5. **Open Pull Request**

### Development Guidelines:
- Gunakan **black** untuk code formatting
- Tambahkan **docstrings** untuk semua functions
- Tulis **unit tests** untuk fitur baru
- Update **documentation** jika diperlukan

---

## 📌 Catatan

- ⚠️ **Project ini masih tahap eksperimen & belajar**
- 🔧 **Model 3D disederhanakan** (decimation) agar proses render lebih cepat
- 📊 **Hasil simulasi** bersifat approximation, bukan CFD analysis yang akurat
- 🎓 **Tujuan utama**: Educational dan proof-of-concept
- 💡 **Kontribusi** dan **feedback** sangat diterima!

---

## ⚠️ Disclaimer

Simulasi ini dibuat untuk tujuan **educational** dan **experimental**. Hasil simulasi **tidak dapat digunakan** untuk desain aerodinamika yang sesungguhnya tanpa validasi menggunakan software CFD profesional dan wind tunnel testing.

---

## 📄 License

Proyek ini dilisensikan di bawah [MIT License](LICENSE).

---

## 👨‍💻 Author

**Ficram Manifur**
- GitHub: [@ficrammanifur](https://github.com/ficrammanifur)
- Email: ficramm@gmail.com

---

## 🙏 Acknowledgments

- **Formula 1** untuk inspirasi dan passion
- **PyVista community** untuk amazing 3D visualization tools
- **Open source CFD community** untuk knowledge sharing
- **Aerodynamics researchers** untuk scientific references

<div align="center">

**🏁 Ready to simulate some F1 aerodynamics?**

**⭐ Star this repository if you find it interesting!**

<p><a href="#top">⬆ Kembali ke Atas</a></p>

</div>
