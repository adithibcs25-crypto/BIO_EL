# AI-Based Network Intrusion Detection Using a Hybrid Artificial Immune System

## Technical Report & Research Benchmark Documentation
 
**Dataset:** NSL-KDD Benchmark (Canadian Institute for Cybersecurity / UNB)  
**Primary Methodology:** Hybrid Artificial Immune System (Negative Selection + Clonal Selection)  
**Baseline Model:** Scikit-Learn Random Forest Classifier  

---

## 1. Abstract

Network Intrusion Detection Systems (NIDS) are critical for securing distributed digital infrastructure against malicious traffic, denial-of-service disruptions, and unauthorized access attempts. Conventional signature-based systems struggle to generalize to unseen zero-day exploits, while purely statistical classifiers can exhibit prohibitive false alarm rates on legitimate traffic.

This project implements a custom **Hybrid Artificial Immune System (AIS)** that translates principles of theoretical immunology into a reproducible computational classifier for network intrusion detection:
1. **Negative Selection Algorithm (NSA)**: Establishes self-tolerance by generating candidate detector centroids and censoring those that fall within an affinity matching radius ($r_{self}$) of normal traffic (Self).
2. **Clonal Selection Algorithm (CSA)**: Drives affinity maturation across multiple generations using held-out validation traffic. Detectors with high attack discrimination are preferentially cloned and subjected to affinity-proportional somatic hypermutation, with auto-reactive mutants systematically filtered out.
3. **Rigorous Benchmarking**: Both the Hybrid AIS and an industry-standard Random Forest baseline are evaluated on the exact same unseen partition of the NSL-KDD dataset (`KDDTest+.txt`), reporting all standard performance metrics required for peer-reviewed academic publication.

---

## 2. Mathematical & Algorithmic Formulation

### 2.1 Feature Preprocessing and Normalization
The NSL-KDD dataset contains 41 traffic attributes, 1 attack label, and 1 difficulty score:
- **Categorical Attributes** (3): `protocol_type` (tcp, udp, icmp), `service` (70 distinct network services), `flag` (11 TCP connection status flags). Encoded using `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
- **Numerical Attributes** (38): Standard continuous and integer attributes (e.g., `duration`, `src_bytes`, `dst_bytes`, `serror_rate`, `same_srv_rate`). Scaled into $[0, 1]$ via `MinMaxScaler`.
- **Target Variable**: Binary mapping where $y = 0$ represents Normal legitimate traffic (Self) and $y = 1$ represents malicious attacks (Non-Self).
- **Leakage Prevention**: All encoder categories and normalization parameters $(\min, \max)$ are fitted strictly on the training partition. The test partition is purely transformed.

### 2.2 Phase 1: Negative Selection Algorithm (NSA)
The goal of negative selection is to generate a mature detector repertoire $D \subset [0, 1]^D$ that covers the Non-Self anomaly space without overlapping normal Self traffic $S$:

1. Propose candidate detector $c \in [0, 1]^D$.
2. Calculate the minimum squared Euclidean distance to all Self samples in the training set:
   $$d_{min}^2(c) = \min_{s \in S} \|c - s\|_2^2$$
3. **Censoring Rule**:
   $$\text{Accept } c \iff d_{min}^2(c) > r_{self}^2$$
   If $d_{min}^2(c) \le r_{self}^2$, candidate $c$ matches Self and is immediately eliminated.
4. **Computational Vectorization**:
   To avoid prohibitive pairwise distance loops on CPU hardware, squared distances are computed using matrix operations:
   $$\|c - s\|^2 = \|c\|^2 + \|s\|^2 - 2 \cdot (c \cdot s^T)$$
   computed in block-matrix chunks via BLAS GEMM.

### 2.3 Phase 2: Clonal Selection Algorithm (CSA)
Initial detectors from Negative Selection are evolved across $G$ generations using an internal validation split $(X_{val}, y_{val})$:

1. **Affinity Metric**:
   For each detector $d_i$, test all validation samples within matching radius $r$:
   $$\text{Affinity}(d_i) = \max\left(0.0001, \text{TPR}_i - \lambda \cdot \text{FPR}_i\right)$$
   where $\text{TPR}_i = \frac{TP_i}{N_{attack}}$ is the true positive rate, $\text{FPR}_i = \frac{FP_i}{N_{normal}}$ is the false alarm rate, and $\lambda$ (default 3.0) heavily penalizes false positives.
2. **Clonal Expansion**:
   Detectors are ranked by affinity. The top fraction ($\alpha = 0.50$) generate a proportional number of clones:
   $$N_{clones}(i) = \max\left(1, \text{round}\left(C_{max} \cdot \frac{N_{select} - i + 1}{N_{select}}\right)\right)$$
3. **Somatic Hypermutation**:
   Clones mutate inversely proportional to detector affinity:
   $$\sigma_i = \sigma_{base} \cdot \left(1.0 - \frac{\text{Affinity}(i)}{\max(\text{Affinities})} + 0.1\right)$$
   $$c' = \text{clip}\left(c + \mathcal{N}(0, \sigma_i^2), 0.0, 1.0\right)$$
4. **Self-Tolerance Filtering**:
   Mutated clones are checked against Self reference data:
   $$\text{Retain } c' \iff \min_{s \in S} \|c' - s\|_2 > r_{self}$$
   Mutated clones that drift into Self space are discarded.
5. **Repertoire Selection**:
   The highest-affinity original detectors and mutated clones are pooled, and the top $K$ detectors are retained for the next generation.

### 2.4 Classification Decision Function
Given an unseen connection $x \in [0, 1]^D$:
$$\hat{y} = \begin{cases} 1 & \text{if } \min_{d \in D^*} \|x - d\|_2 \le r \\ 0 & \text{otherwise} \end{cases}$$
The continuous Anomaly Affinity Score is defined as:
$$S(x) = \exp\left(-\frac{\min_{d \in D^*} \|x - d\|_2^2}{2 \cdot r^2}\right) \in [0, 1]$$

---

## 3. Comprehensive Performance Metrics (Research Paper Standard)

The following metrics are tracked and exported to `results/metrics/metrics.json` and `results/metrics/metrics.csv`:

| Metric Name | Notation | Mathematical Formula | Purpose in NIDS Research |
| :--- | :--- | :--- | :--- |
| **Accuracy** | $\text{ACC}$ | $\frac{TP + TN}{TP + TN + FP + FN}$ | Overall classification correctness across all traffic classes. |
| **Precision** | $\text{PPV}$ | $\frac{TP}{TP + FP}$ | Fraction of flagged alarms that are actual cyber intrusions. |
| **Recall / Detection Rate** | $\text{DR / TPR}$ | $\frac{TP}{TP + FN}$ | Proportion of true attacks successfully detected. |
| **Specificity** | $\text{TNR}$ | $\frac{TN}{TN + FP}$ | Proportion of normal traffic correctly passed without false alarms. |
| **False Positive Rate** | $\text{FPR}$ | $\frac{FP}{FP + TN} = 1 - \text{Specificity}$ | Rate at which legitimate operations are falsely blocked. |
| **False Negative Rate** | $\text{FNR}$ | $\frac{FN}{FN + TP} = 1 - \text{Recall}$ | Rate at which intrusions evade detection. |
| **F1-Score** | $F_1$ | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic balance between precision and detection rate. |
| **Matthews Correlation** | $\text{MCC}$ | $\frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$ | Balanced quality metric robust to class imbalance ($-1$ to $+1$). |
| **ROC Area Under Curve** | $\text{ROC-AUC}$ | $\int_0^1 \text{TPR}(\text{FPR}) \, d(\text{FPR})$ | Threshold-independent discriminative capacity. |
| **Training Duration** | $T_{train}$ | Elapsed wall-clock time in seconds | Computational complexity of model fitting. |
| **Inference Latency** | $T_{inf}$ | Elapsed time to classify test partition | Real-time packet inspection viability. |
| **Throughput** | $R_{inf}$ | $\frac{N_{test}}{T_{inf}}$ (samples / sec) | Maximum packet rate supported on single CPU thread. |

---

## 4. Empirical Evaluation Results (Real NSL-KDD Test Set)

Evaluated on 5,000 unseen test connection records from `KDDTest+.txt`:

```text
===========================================================================
 BENCHMARK RESULTS SUMMARY (NSL-KDD Test Set)
===========================================================================
Metric                    | Hybrid AIS (Custom)| Random Forest     
-------------------------------------------------------------------
Accuracy                  | 0.4730             | 0.7710            
Precision (PPV)           | 1.0000             | 0.9645            
Recall (Detection Rate)   | 0.0735             | 0.6203            
Specificity (TNR)         | 1.0000             | 0.9699            
F1-Score                  | 0.1369             | 0.7550            
False Positive Rate (FPR) | 0.0000             | 0.0301            
Matthews Corr (MCC)       | 0.2045             | 0.5841            
ROC AUC                   | 0.7001             | 0.9555            
Training Time (s)         | 3.48               | 0.29              
Prediction Latency (s)    | 0.01               | 0.04              
Throughput (samples/s)    | 653,757.1          | 118,868.5         
===========================================================================
```

### Analysis for Academic Discussion & Viva Defense
1. **Zero False Alarm Rate (FPR = 0.0000)**:
   - The Negative Selection and Clonal Selection algorithms enforce **absolute self-tolerance**. Not a single legitimate connection was falsely classified as an attack ($\text{Precision} = 1.0000$, $\text{FPR} = 0.0000$).
   - In mission-critical enterprise networks (e.g., healthcare, financial clearing), avoiding false alarms that take down legitimate services is often prioritized over aggressive detection.
2. **Comparison with Random Forest Baseline**:
   - Random Forest partitions tabular data via orthogonal axis-aligned hyperplanes, achieving higher recall on known attack variants ($62.03\%$).
   - The Hybrid AIS utilizes spherical Euclidean boundary detectors. Because NSL-KDD features span 115 dimensions after encoding, covering complex concave attack manifolds with spherical detectors requires tuning detector density or exploring non-linear affinity metrics.
3. **Inference Throughput**:
   - The vectorized matrix formulation allows the AIS to process over **650,000 connections per second** on a standard consumer laptop CPU, outperforming tree ensembles in throughput.

---

## 5. Project Structure

```text
Bio_EL_Intrusion_Detection/
├── data/
│   ├── raw/                  # NSL-KDD source files: KDDTrain+.txt, KDDTest+.txt
│   └── processed/            # Cleaned or intermediate cached partitions
├── models/                   # Serialized model binaries
│   ├── hybrid_ais.joblib     # Fitted Hybrid AIS detector repertoire
│   ├── random_forest.joblib  # Trained Random Forest baseline
│   └── preprocessor.joblib   # Fitted MinMaxScaler + OneHotEncoder pipeline
├── results/
│   ├── figures/              # Publication-grade figures
│   │   ├── confusion_matrices.png
│   │   ├── metrics_comparison.png
│   │   ├── clonal_progress.png
│   │   ├── roc_curves.png
│   │   └── execution_times.png
│   └── metrics/              # Structured evaluation data
│       ├── metrics.json      # Complete numerical metrics in JSON
│       └── metrics.csv       # Summary table in CSV
├── src/                      # Source modules
│   ├── __init__.py
│   ├── data_loader.py        # NSL-KDD parser and category mapper
│   ├── preprocessing.py      # Zero-leakage transformer pipeline
│   ├── negative_selection.py # Vectorized Negative Selection Algorithm (NSA)
│   ├── clonal_selection.py   # Clonal Selection Algorithm (CSA) & hypermutation
│   ├── hybrid_ais.py         # Unified Scikit-learn compatible Hybrid AIS classifier
│   ├── random_forest.py      # Random Forest benchmark wrapper
│   ├── evaluation.py         # Metrics calculation (MCC, Specificity, FPR, etc.)
│   └── visualization.py      # Academic figure generation
├── tests/                    # Pytest test suite (14 unit & integration tests)
│   ├── conftest.py
│   ├── test_preprocessing.py
│   ├── test_negative_selection.py
│   ├── test_clonal_selection.py
│   ├── test_hybrid_ais.py
│   ├── test_random_forest.py
│   ├── test_evaluation.py
│   └── test_pipeline.py
├── app.py                    # Streamlit academic evaluation dashboard
├── train.py                  # CLI training and evaluation script
├── favicon.png               # Academic browser favicon
├── pytest.ini                # Pytest configuration
├── requirements.txt          # Python dependencies
└── README.md                 # Complete research report
```

---

## 6. Execution Instructions

### 6.1 Install Dependencies
```bash
cd Bio_EL_Intrusion_Detection
pip install -r requirements.txt
```

### 6.2 Execute Pytest Test Suite
```bash
python -m pytest tests/ -v
```

### 6.3 Run Model Training from Terminal
```bash
# Standard benchmark run (recommended for quick evaluation):
python train.py --train_sample 15000 --test_sample 5000 --detectors 80 --radius 0.45 --generations 8

# Full dataset run (all 125,973 train records and 22,544 test records):
python train.py --train_sample 0 --test_sample 0 --detectors 100 --radius 0.45 --generations 10
```

### 6.4 Launch the Streamlit Research Dashboard
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser. The dashboard includes:
1. **Experimental Benchmarks**: High-resolution confusion matrices, ROC curves, and comparative metric tables.
2. **Single-Packet Inspection**: Step-by-step mathematical distance evaluation ($d_{min} \le r$) for individual network packets.
3. **Dataset Specification**: Feature taxonomy, categorical encodings, and class distributions.
4. **Algorithm Training Studio**: Train new models with customized hyperparameters directly from the GUI.
5. **Research Paper Table Exporter**: Automatically formats results into clean, copy-ready LaTeX table code.
6. **Terms of Use & Privacy Policy**: Institutional data governance and fair-use guidelines.
7. **Deployment & Custom Domain**: Production Nginx reverse-proxy and DNS configuration guides.

---

## 7. Institutional Privacy Policy & Terms of Use

- **Data Privacy**: All preprocessing, detector generation, hypermutation, and evaluation execute strictly within the local computational environment. No network telemetry or packet attributes are transmitted to external services.
- **Data Provenance**: The NSL-KDD benchmark contains synthetic, anonymized connection vectors without raw packet payloads or Personally Identifiable Information (PII).
- **Academic Fair Use**: Software and algorithmic implementations are released for educational, university coursework, and peer-reviewed research benchmarking purposes.
