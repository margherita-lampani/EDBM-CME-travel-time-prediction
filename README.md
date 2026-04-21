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

<div style="overflow-x: auto;">

<table>
  <thead>
    <tr>
      <th>Start Date</th>
      <th>Arrival Date</th>
      <th>Transit Time [h] (t)</th>
      <th>Initial Speed [km/s] (v₀)</th>
      <th>Arrival Speed [km/s] (v)</th>
      <th>Mass [g] (m)</th>
      <th>Impact Area [sr] (A)</th>
      <th>SW Density [cm⁻³] (ρ)</th>
      <th>SW Speed [km/s] (w)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>1997-08-30 06:56:41</td><td>1997-09-03 13:00:00</td><td>102.06</td><td>557.61</td><td>410</td><td>1.7×10¹⁵</td><td>0.7854</td><td>5.718</td><td>423.5</td></tr>
    <tr><td>1997-09-28 06:02:03</td><td>1997-10-01 16:00:00</td><td>81.97</td><td>603.39</td><td>450</td><td>4.0×10¹⁵</td><td>0.7854</td><td>7.341</td><td>471.5</td></tr>
    <tr><td>1997-12-06 18:05:37</td><td>1997-12-10 18:00:00</td><td>95.91</td><td>682.83</td><td>350</td><td>2.0×10¹⁶</td><td>0.7854</td><td>4.913</td><td>370.6</td></tr>
    <tr><td>1998-01-25 17:41:60</td><td>1998-01-29 20:00:00</td><td>98.30</td><td>1319.9</td><td>380</td><td>1.1×10¹⁶</td><td>1.1519</td><td>6.406</td><td>405.2</td></tr>
    <tr><td>1998-05-02 16:17:36</td><td>1998-05-04 10:00:00</td><td>41.71</td><td>1358.3</td><td>550</td><td>7.7×10¹⁵</td><td>1.1519</td><td>4.232</td><td>614.8</td></tr>
    <tr><td>1999-04-13 10:02:37</td><td>1999-04-16 18:00:00</td><td>79.96</td><td>466.56</td><td>410</td><td>1.4×10¹⁵</td><td>0.5585</td><td>3.088</td><td>418.8</td></tr>
    <tr><td>1999-07-04 00:47:21</td><td>1999-07-06 21:00:00</td><td>68.21</td><td>826.23</td><td>460</td><td>8.4×10¹⁵</td><td>0.7854</td><td>2.419</td><td>473.7</td></tr>
    <tr><td>1999-07-24 03:32:04</td><td>1999-07-27 17:00:00</td><td>85.47</td><td>512.63</td><td>390</td><td>7.5×10¹⁵</td><td>0.7854</td><td>4.249</td><td>464.7</td></tr>
    <tr><td>2000-02-12 06:54:10</td><td>2000-02-14 12:00:00</td><td>53.10</td><td>1286.9</td><td>520</td><td>1.6×10¹⁵</td><td>1.1519</td><td>16.425</td><td>593.7</td></tr>
    <tr><td>2000-07-17 11:58:46</td><td>2000-07-20 01:00:00</td><td>61.02</td><td>1431.9</td><td>530</td><td>7.5×10¹⁵</td><td>1.1519</td><td>1.792</td><td>673.3</td></tr>
    <tr><td>2000-07-23 09:24:26</td><td>2000-07-27 02:00:00</td><td>88.59</td><td>792.55</td><td>360</td><td>3.1×10¹⁵</td><td>0.7854</td><td>4.864</td><td>420.2</td></tr>
    <tr><td>2000-08-29 20:44:59</td><td>2000-09-02 22:00:00</td><td>97.25</td><td>1200.7</td><td>420</td><td>1.9×10¹⁵</td><td>1.1519</td><td>3.197</td><td>599.8</td></tr>
    <tr><td>2000-10-25 12:49:10</td><td>2000-10-28 21:00:00</td><td>80.18</td><td>1196.1</td><td>380</td><td>1.7×10¹⁶</td><td>1.1519</td><td>3.639</td><td>398.8</td></tr>
    <tr><td>2001-09-28 11:08:01</td><td>2001-10-01 08:00:00</td><td>68.87</td><td>1318.1</td><td>490</td><td>2.0×10¹⁶</td><td>1.1519</td><td>2.048</td><td>513.5</td></tr>
    <tr><td>2001-09-29 14:50:15</td><td>2001-10-02 04:00:00</td><td>61.16</td><td>922.6</td><td>490</td><td>1.4×10¹⁵</td><td>1.1519</td><td>3.342</td><td>656.8</td></tr>
    <tr><td>2001-10-22 22:03:17</td><td>2001-10-27 03:00:00</td><td>100.95</td><td>810.11</td><td>420</td><td>1.0×10¹⁵</td><td>0.7854</td><td>11.035</td><td>520.9</td></tr>
    <tr><td>2001-10-25 17:45:21</td><td>2001-10-29 22:00:00</td><td>100.24</td><td>1343.1</td><td>360</td><td>4.2×10¹⁵</td><td>1.1519</td><td>13.567</td><td>443.3</td></tr>
    <tr><td>2003-05-28 02:31:26</td><td>2003-05-30 02:00:00</td><td>47.48</td><td>1744.5</td><td>600</td><td>1.3×10¹⁶</td><td>1.1519</td><td>6.763</td><td>651.5</td></tr>
    <tr><td>2003-06-14 05:01:34</td><td>2003-06-17 10:00:00</td><td>76.97</td><td>993.26</td><td>480</td><td>4.4×10¹⁵</td><td>1.1519</td><td>8.957</td><td>482.3</td></tr>
    <tr><td>2005-01-05 18:07:09</td><td>2005-01-08 21:00:00</td><td>74.88</td><td>1225.4</td><td>460</td><td>4.5×10¹⁵</td><td>1.1519</td><td>2.715</td><td>647.9</td></tr>
    <tr><td>2005-01-13 20:33:41</td><td>2005-01-16 14:00:00</td><td>65.44</td><td>1043.7</td><td>520</td><td>4.2×10¹⁴</td><td>1.1519</td><td>2.372</td><td>731.5</td></tr>
    <tr><td>2006-12-15 01:00:50</td><td>2006-12-17 00:00:00</td><td>46.99</td><td>1194.4</td><td>580</td><td>7.5×10¹⁵</td><td>1.1519</td><td>10.808</td><td>776.0</td></tr>
    <tr><td>2008-12-12 21:38:06</td><td>2008-12-17 03:00:00</td><td>101.36</td><td>422.42</td><td>350</td><td>8.1×10¹⁴</td><td>0.5585</td><td>6.025</td><td>371.6</td></tr>
    <tr><td>2011-06-14 12:12:07</td><td>2011-06-17 05:00:00</td><td>64.80</td><td>941.28</td><td>500</td><td>1.5×10¹⁶</td><td>1.1519</td><td>4.951</td><td>524.9</td></tr>
    <tr><td>2011-08-02 09:26:39</td><td>2011-08-05 05:00:00</td><td>67.56</td><td>990.79</td><td>430</td><td>5.1×10¹⁵</td><td>1.1519</td><td>1.635</td><td>494.8</td></tr>
    <tr><td>2011-09-14 03:08:49</td><td>2011-09-17 14:00:00</td><td>82.85</td><td>896.62</td><td>430</td><td>2.5×10¹⁵</td><td>0.7854</td><td>2.455</td><td>540.1</td></tr>
    <tr><td>2011-10-27 16:13:00</td><td>2011-11-02 01:00:00</td><td>128.78</td><td>724.61</td><td>380</td><td>5.8×10¹⁵</td><td>0.7854</td><td>1.995</td><td>426.5</td></tr>
    <tr><td>2012-01-18 19:03:46</td><td>2012-01-21 06:00:00</td><td>58.94</td><td>465.94</td><td>320</td><td>2.2×10¹⁵</td><td>0.5585</td><td>5.182</td><td>396.1</td></tr>
    <tr><td>2012-02-24 07:14:15</td><td>2012-02-27 19:00:00</td><td>83.76</td><td>995.19</td><td>440</td><td>6.3×10¹⁵</td><td>1.1519</td><td>3.902</td><td>445.6</td></tr>
    <tr><td>2012-05-12 01:49:45</td><td>2012-05-16 16:00:00</td><td>110.17</td><td>1493.5</td><td>370</td><td>4.6×10¹⁵</td><td>1.1519</td><td>1.658</td><td>555.0</td></tr>
    <tr><td>2012-07-02 10:39:27</td><td>2012-07-05 00:00:00</td><td>61.34</td><td>1350.5</td><td>470</td><td>6.0×10¹⁵</td><td>1.1519</td><td>3.025</td><td>614.5</td></tr>
    <tr><td>2012-07-04 20:39:47</td><td>2012-07-09 00:00:00</td><td>99.34</td><td>953.92</td><td>410</td><td>3.4×10¹⁴</td><td>1.1519</td><td>2.905</td><td>488.2</td></tr>
    <tr><td>2012-09-28 02:45:48</td><td>2012-10-01 00:00:00</td><td>69.24</td><td>999.16</td><td>370</td><td>9.2×10¹⁵</td><td>1.1519</td><td>2.976</td><td>387.7</td></tr>
    <tr><td>2012-11-20 15:38:23</td><td>2012-11-24 12:00:00</td><td>92.36</td><td>986.34</td><td>380</td><td>8.4×10¹⁵</td><td>1.1519</td><td>6.448</td><td>391.8</td></tr>
    <tr><td>2014-02-12 13:40:45</td><td>2014-02-16 05:00:00</td><td>87.32</td><td>421.23</td><td>380</td><td>5.0×10¹⁵</td><td>0.5585</td><td>1.442</td><td>420.6</td></tr>
    <tr><td>2014-04-02 15:23:06</td><td>2014-04-05 22:00:00</td><td>78.61</td><td>1815.7</td><td>380</td><td>1.4×10¹⁶</td><td>1.1519</td><td>4.056</td><td>433.6</td></tr>
    <tr><td>2014-09-12 21:06:02</td><td>2014-09-17 02:00:00</td><td>100.90</td><td>1308.9</td><td>310</td><td>1.2×10¹⁶</td><td>1.1519</td><td>17.653</td><td>659.0</td></tr>
    <tr><td>2015-11-04 17:54:55</td><td>2015-11-07 06:00:00</td><td>60.08</td><td>948.3</td><td>500</td><td>6.6×10¹⁵</td><td>1.1519</td><td>1.345</td><td>661.5</td></tr>
    <tr><td>2015-12-16 12:04:07</td><td>2015-12-20 03:00:00</td><td>86.93</td><td>1079.9</td><td>400</td><td>2.3×10¹⁵</td><td>1.1519</td><td>1.931</td><td>541.3</td></tr>
    <tr><td>2015-12-28 14:25:12</td><td>2015-12-31 17:00:00</td><td>74.58</td><td>1520.5</td><td>440</td><td>1.9×10¹⁶</td><td>1.1519</td><td>3.179</td><td>455.0</td></tr>
    <tr><td>2016-07-17 17:45:18</td><td>2016-07-20 07:00:00</td><td>61.25</td><td>510.31</td><td>440</td><td>6.1×10¹⁴</td><td>0.7854</td><td>2.130</td><td>478.8</td></tr>
  </tbody>
</table>

</div>

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

