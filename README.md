# EDBM-CME-travel-time-prediction
EDBM‑CME‑travel‑time‑prediction provides neural‑network travel‑time forecasts for CMEs using the Extended Drag‑Based Model (EDBM), plus a multiclass classifier to identify different CME dynamical regimes. It includes preprocessing, training scripts, and reproducible analysis tools.
# EDBM CME Transit Time Prediction – Code Package

---

## Repository Structure

```
CME_FINAL/
├── utils/
│   ├── data_loading.py      # Data I/O, unit conversion, case classification
│   ├── optimization.py      # EDBM parameter optimization (per propagation case)
│   └── augmentation.py      # Data augmentation and stratified splitting
├── models/
│   ├── transit_time_nn.py   # Physics-informed neural network (Case Cross wind -)
│   └── classification.py    # Logistic regression classifiers (multi-class)
├── run_optimization.py      # Script 1: optimize EDBM parameter 'a' per case
├── run_transit_time.py      # Script 2: train the transit time neural network
├── run_classification.py    # Script 3: train propagation-case classifiers
└── visualization_results.ipynb  # Notebook: load results CSV and generate all figures
```

---

## Data Requirements

| File | Description |
|---|---|
| `Data/ICME_complete_dataset_rev.csv` | ICME observational catalog |
| `Results/0_results_100_best.csv` | Pre-computed neural network results (visualization only) |

---

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
for Case Cross wind (-) events.  Results are saved to `results.csv`.

Requires: **TensorFlow 2.12**.

### Regime classification classification – Train propagation case classifiers

```bash
python run_classification.py
```

Trains logistic regression classifiers for 6-class classification of all propagation regimes.

### Visualization (no computation required)

Open and run `visualization_results.ipynb`.  
It reads `0_results_100_best.csv` and generates all figures from the paper.


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
