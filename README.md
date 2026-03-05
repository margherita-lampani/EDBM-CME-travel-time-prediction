# EDBM-ML-CME-Arrival-Time-Prediction
EDBM‑CME‑travel‑time‑prediction provides neural‑network travel‑time forecasts for CMEs using the Extended Drag‑Based Model (EDBM), plus a multiclass classifier to identify different CME dynamical regimes. It includes preprocessing, training scripts, and reproducible analysis tools.
# EDBM ML CME Arrival Time Prediction – Code Package

---

## Repository Structure

```
CME_FINAL/
├── utils/
│   ├── data_loading.py      # Data I/O, unit conversion, case classification
│   ├── optimization.py      # EDBM parameter 'a' optimization (per Case Cross-w (↘))
│   └── augmentation.py      # Data augmentation and stratified splitting
├── models/
│   ├── transit_time_nn.py   # Physics-informed neural network (Case Case Cross-w (↘))
│   └── classification.py    # Logistic regression classifiers (multi-class)
├── run_optimization.py      # Script 1: optimize EDBM parameter 'a' per Case Cross-w (↘)
├── run_transit_time.py      # Script 2: train the transit time neural network
├── run_classification.py    # Script 3: train propagation-case classifiers
└── results_visualization.ipynb  # Notebook: load results CSV and generate all figures
```

---

## Data Requirements

| File | Description |
|---|---|
| `Data/ICME_complete_dataset_rev.csv` | ICME observational catalog |
| `Results/final_results.csv` | Pre-computed neural network results (visualization only) |

---
### Data Sources

The ICME dataset used in this project is based on the catalog presented in **Napoletano et al. (2022)**  
(https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021SW002925), which combines events from the Richardson & Cane ICME list and the SOHO/LASCO CME catalog.  
It includes parameters relevant for drag-based CME models, such as initial speed, arrival speed, travel time, angular width, and CME mass.

Background solar wind speed and density follow the values adopted in **Guastavino et al. (2023)**  
(https://iopscience.iop.org/article/10.3847/1538-4357/ace62d/meta), derived from CELIAS measurements.  
These values are used to classify CME events into different EDBM regimes.

## How to Run

### Step 1 – Optimize EDBM acceleration parameter

```bash
python run_optimization.py
```

Reads the CSV file, classifies events into the 6 propagation cases,
and solves the analytical drag-based model equation for each event to find
the optimal acceleration parameter *a*.

### Step 2 – Train the transit time neural network

```bash
python run_transit_time.py
```

Trains the physics-informed neural network over 25 independent realizations
for Case Cross-w (↘) events.  Results are saved to `Results/results.csv`.

Requires: **TensorFlow 2.12**.

### Regime classification classification – Train propagation case classifiers

```bash
python run_classification.py
```

Trains logistic regression classifiers for 6-class classification of all propagation regimes.

### Visualization (no computation required)

Open and run `visualization_results.ipynb`.  
It reads `final_results.csv` and generates all figures from the paper.


---

## Dependencies (requirements.txt)

```
numpy
pandas
scipy
scikit-learn
tensorflow==2.12.0
matplotlib
seaborn
```

---

## Citation

If you use this code or the associated methodology in your research, please cite the following works:

* **Lampani et al.** – *Neural-network CME transit time prediction using the Extended Drag-Based Model (EDBM)*
  https://arxiv.org/abs/2512.19492

* **Rossi et al. (2025)** – *Physics-informed drag-based modeling of CME propagation*
  https://www.aanda.org/articles/aa/full_html/2025/02/aa52288-24/aa52288-24.html

