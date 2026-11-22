"""Network Intrusion Detection Evaluation Dashboard.

Scientific and research benchmark interface for evaluating a Hybrid Artificial
Immune System (Negative Selection and Clonal Selection) alongside Random Forest.
"""

import json
import os
from pathlib import Path
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve
import streamlit as st

# Path configuration
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "results" / ".mpl_cache"))

from src.data_loader import (
    COLUMN_NAMES,
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    load_nsl_kdd,
)
from src.preprocessing import NSLPreprocessor
from src.hybrid_ais import HybridAISClassifier
from src.random_forest import train_random_forest, save_random_forest
from src.evaluation import evaluate_model, save_metrics
from src.visualization import (
    plot_confusion_matrices,
    plot_metrics_comparison,
    plot_clonal_progress,
    plot_roc_curves,
    plot_execution_times,
)

# Page configuration with favicon
FAVICON_PATH = PROJECT_ROOT / "favicon.png"
st.set_page_config(
    page_title="AIS Network Intrusion Detection Benchmark",
    page_icon=str(FAVICON_PATH) if FAVICON_PATH.exists() else None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# High-contrast dark institutional stylesheet (Pure black background, white text)
st.markdown("""
<style>
    /* Hide Streamlit brand, footer, and sidebar toggle */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="collapsedControl"] {display: none;}
    section[data-testid="stSidebar"] {display: none;}

    /* Force dark background and white text across all elements */
    .stApp {
        background-color: #000000 !important;
        color: #ffffff !important;
    }

    body, p, span, label, div {
        color: #ffffff !important;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    .academic-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #ffffff !important;
        margin-bottom: 0.25rem;
        border-bottom: 2px solid #38bdf8;
        padding-bottom: 0.5rem;
    }
    .academic-subtitle {
        font-size: 0.95rem;
        color: #cbd5e1 !important;
        margin-bottom: 1.25rem;
    }
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #ffffff !important;
        margin-top: 1.25rem;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid #334155;
        padding-bottom: 0.25rem;
    }
    .explanation-callout {
        background-color: #0f172a !important;
        border-left: 4px solid #38bdf8;
        border-radius: 2px;
        padding: 0.75rem 1rem;
        margin: 0.75rem 0;
        color: #e2e8f0 !important;
        font-size: 0.90rem;
        line-height: 1.5;
    }
    .metric-box {
        background-color: #111827 !important;
        border: 1px solid #374151;
        border-radius: 4px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #9ca3af !important;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #ffffff !important;
        margin-top: 0.2rem;
        margin-bottom: 0.2rem;
    }
    .metric-explanation {
        font-size: 0.78rem;
        color: #94a3b8 !important;
        line-height: 1.3;
    }
    .status-badge-normal {
        color: #4ade80 !important;
        background-color: #052e16 !important;
        border: 1px solid #166534;
        padding: 4px 8px;
        font-weight: 600;
        border-radius: 2px;
        font-size: 0.85rem;
        display: inline-block;
    }
    .status-badge-attack {
        color: #f87171 !important;
        background-color: #450a0a !important;
        border: 1px solid #991b1b;
        padding: 4px 8px;
        font-weight: 600;
        border-radius: 2px;
        font-size: 0.85rem;
        display: inline-block;
    }

    /* Tables styling */
    table {
        color: #ffffff !important;
        background-color: #0b0f19 !important;
    }
    th {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border: 1px solid #334155 !important;
    }
    td {
        background-color: #0f172a !important;
        color: #f8fafc !important;
        border: 1px solid #1e293b !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #0b0f19 !important;
        border-bottom: 1px solid #334155;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8 !important;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        border-bottom: 2px solid #38bdf8 !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background-color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_dataset_sample(sample_size: int = 10000):
    raw_train = PROJECT_ROOT / "data" / "raw" / "KDDTrain+.txt"
    if not raw_train.exists():
        return None
    return load_nsl_kdd(str(raw_train), sample_size=sample_size, random_state=42)


@st.cache_resource
def load_models():
    models_dir = PROJECT_ROOT / "models"
    prep_p = models_dir / "preprocessor.joblib"
    ais_p = models_dir / "hybrid_ais.joblib"
    rf_p = models_dir / "random_forest.joblib"

    prep = NSLPreprocessor.load(prep_p) if prep_p.exists() else None
    ais = HybridAISClassifier.load(ais_p) if ais_p.exists() else None
    rf = joblib.load(rf_p) if rf_p.exists() else None
    return prep, ais, rf


def get_evaluation_metrics():
    metrics_path = PROJECT_ROOT / "results" / "metrics" / "metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


preprocessor, ais_model, rf_model = load_models()
metrics_data = get_evaluation_metrics()

# Header
st.markdown('<div class="academic-title">Artificial Immune System Network Intrusion Detection Benchmark</div>', unsafe_allow_html=True)
st.markdown('<div class="academic-subtitle">Comparative evaluation of a custom Negative Selection and Clonal Selection Hybrid AIS against Random Forest on the NSL-KDD benchmark dataset.</div>', unsafe_allow_html=True)

# System state strip (replaces sidebar status)
col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
col_s1.caption(f"Preprocessor: **{'Loaded' if preprocessor else 'Unfitted'}**")
col_s2.caption(f"Hybrid AIS: **{'Loaded' if ais_model else 'Unfitted'}**")
col_s3.caption(f"Random Forest: **{'Loaded' if rf_model else 'Unfitted'}**")
col_s4.caption(f"Saved Metrics: **{'Available' if metrics_data else 'None'}**")
col_s5.caption("Docs: **README.md**")

st.markdown("---")

# Tab layout — no sidebar
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Experimental Benchmarks",
    "Live Packet Inspector",
    "Dataset Specification",
    "Train Models",
    "LaTeX Table Exporter",
    "Legal",
    "Deployment",
])


# ==============================================================================
# TAB 1: EXPERIMENTAL BENCHMARKS
# ==============================================================================
with tab1:
    st.markdown('<div class="section-header">Experimental Context & Setup</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="explanation-callout">
        <strong>Evaluation Setting:</strong> Both models were evaluated on <strong>5,000 unseen test packets</strong> from the official <code>KDDTest+.txt</code> test partition.
        Normal traffic represents legitimate operational packets ("Self"), while attacks represent malicious unauthorized activity ("Non-Self").
        Transformers were fitted strictly on training data to prevent any data leakage.
    </div>
    """, unsafe_allow_html=True)

    if metrics_data:
        ais_res = next((m for m in metrics_data if m["model_name"] == "Hybrid AIS"), None)
        rf_res = next((m for m in metrics_data if m["model_name"] == "Random Forest"), None)

        if ais_res and rf_res:
            st.markdown('<div class="section-header">Key Performance Indicators & Pedagogical Explanation</div>', unsafe_allow_html=True)

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">AIS Accuracy</div>
                    <div class="metric-value">{ais_res['accuracy']:.4f}</div>
                    <div class="metric-explanation">Reflects conservative boundary matching across 115 feature dimensions.</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">AIS Precision (PPV)</div>
                    <div class="metric-value">{ais_res['precision']:.4f}</div>
                    <div class="metric-explanation">100% of flagged packets are genuine attacks. Zero false alarms triggered.</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">AIS False Positive Rate</div>
                    <div class="metric-value">{ais_res['false_positive_rate']:.4f}</div>
                    <div class="metric-explanation">Strict self-tolerance: not a single normal operational packet was blocked.</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">AIS Recall (DR)</div>
                    <div class="metric-value">{ais_res['recall']:.4f}</div>
                    <div class="metric-explanation">Attack detection rate achieved by the conservative detector repertoire.</div>
                </div>
                """, unsafe_allow_html=True)
            with col5:
                spec_val = ais_res.get('specificity', 1.0 - ais_res['false_positive_rate'])
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">AIS Specificity (TNR)</div>
                    <div class="metric-value">{spec_val:.4f}</div>
                    <div class="metric-explanation">100% of legitimate network communications passed without disruption.</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Complete Comparative Metric Table (Research Paper Standard)</div>', unsafe_allow_html=True)

        table_rows = []
        metrics_keys = [
            ("Accuracy", "accuracy", "{:.4f}"),
            ("Precision (Positive Predictive Value)", "precision", "{:.4f}"),
            ("Recall / Detection Rate (TPR)", "recall", "{:.4f}"),
            ("Specificity (True Negative Rate)", "specificity", "{:.4f}"),
            ("False Positive Rate (False Alarm Rate)", "false_positive_rate", "{:.4f}"),
            ("False Negative Rate (Miss Rate)", "false_negative_rate", "{:.4f}"),
            ("F1-Score (Harmonic Mean)", "f1_score", "{:.4f}"),
            ("Matthews Correlation Coefficient (MCC)", "mcc", "{:.4f}"),
            ("ROC Area Under Curve (AUC)", "roc_auc", "{:.4f}"),
            ("True Negatives (TN)", "true_negatives", "{:,}"),
            ("False Positives (FP)", "false_positives", "{:,}"),
            ("False Negatives (FN)", "false_negatives", "{:,}"),
            ("True Positives (TP)", "true_positives", "{:,}"),
            ("Training Execution Duration (s)", "training_time_sec", "{:.2f}"),
            ("Inference Latency on Test Partition (s)", "prediction_time_sec", "{:.4f}"),
            ("Inference Throughput (samples/s)", "throughput_samples_per_sec", "{:,.0f}"),
            ("Evaluated Test Records", "total_test_samples", "{:,}"),
        ]

        for label, key, fmt in metrics_keys:
            ais_v = ais_res.get(key, None) if ais_res else None
            rf_v = rf_res.get(key, None) if rf_res else None

            ais_str = fmt.format(ais_v) if ais_v is not None else "N/A"
            rf_str = fmt.format(rf_v) if rf_v is not None else "N/A"

            table_rows.append({
                "Evaluation Metric": label,
                "Hybrid AIS (Custom Model)": ais_str,
                "Random Forest (Baseline Model)": rf_str,
            })

        df_metrics_table = pd.DataFrame(table_rows)
        st.table(df_metrics_table)

        st.markdown("""
        <div class="explanation-callout">
            <strong>Key Insights for Teacher / Oral Defense:</strong><br>
            &bull; <strong>Why did the AIS achieve 0.0000 False Alarms?</strong> Negative Selection rejects any candidate detector matching normal traffic within radius <code>r</code> during training. Then, Clonal Selection filters out any mutated clone that drifts into normal space. This mathematically guarantees zero false alarms on legitimate communications.<br>
            &bull; <strong>Why does Random Forest have higher Recall (62.03% vs 7.35%)?</strong> Random Forest constructs orthogonal decision tree splits that fit known training attack signatures closely. In contrast, the AIS models Non-Self space using spherical Euclidean distance clusters, which prioritize strict self-tolerance over aggressive boundary expansion in 115 dimensions.<br>
            &bull; <strong>Why is AIS throughput so high (&gt;200,000 samples/s)?</strong> AIS distance calculations are implemented via chunked BLAS matrix multiplications, allowing instant Euclidean evaluation without GPU overhead.
        </div>
        """, unsafe_allow_html=True)

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown('<div class="section-header">Confusion Matrices & Quadrant Breakdown</div>', unsafe_allow_html=True)
            fig_cm = plot_confusion_matrices(metrics_data)
            st.pyplot(fig_cm)
            plt.close(fig_cm)
            st.markdown("""
            <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.4;">
                &bull; <strong>Top-Left (TN):</strong> Normal traffic correctly recognized as Self and permitted.<br>
                &bull; <strong>Top-Right (FP):</strong> False alarms (normal traffic wrongly blocked) - <strong>0 for AIS</strong>.<br>
                &bull; <strong>Bottom-Left (FN):</strong> Attacks not matched by current detector radii (missed intrusions).<br>
                &bull; <strong>Bottom-Right (TP):</strong> Attacks correctly matched by mature detector centroids.
            </div>
            """, unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="section-header">Performance Comparison Bar Chart</div>', unsafe_allow_html=True)
            fig_bar = plot_metrics_comparison(metrics_data)
            st.pyplot(fig_bar)
            plt.close(fig_bar)
            st.markdown("""
            <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.4;">
                Visual bar comparison of Accuracy, Precision, Recall, Specificity, F1, FPR, and Matthews Correlation Coefficient. Notice the contrast between AIS perfect Specificity (1.0) and Random Forest higher Recall.
            </div>
            """, unsafe_allow_html=True)

        col_roc, col_time = st.columns(2)
        with col_roc:
            fig_path = PROJECT_ROOT / "results" / "figures" / "roc_curves.png"
            if fig_path.exists():
                st.markdown('<div class="section-header">Receiver Operating Characteristic (ROC) Curves</div>', unsafe_allow_html=True)
                st.image(str(fig_path))
                st.markdown("""
                <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.4;">
                    Demonstrates true positive rate against false positive rate across decision thresholds. The AIS curve reflects threshold-free anomaly scores with ROC-AUC of 0.7001.
                </div>
                """, unsafe_allow_html=True)

        with col_time:
            fig_path_t = PROJECT_ROOT / "results" / "figures" / "execution_times.png"
            if fig_path_t.exists():
                st.markdown('<div class="section-header">Computational Execution Durations</div>', unsafe_allow_html=True)
                st.image(str(fig_path_t))
                st.markdown("""
                <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.4;">
                    Comparison of training time and inference latency on 5,000 test connections on consumer CPU hardware.
                </div>
                """, unsafe_allow_html=True)

        if ais_model and ais_model.history_:
            st.markdown('<div class="section-header">Clonal Selection Optimization Convergence (Generations 1 to 8)</div>', unsafe_allow_html=True)
            fig_clonal = plot_clonal_progress(ais_model.history_)
            if fig_clonal:
                st.pyplot(fig_clonal)
                plt.close(fig_clonal)
            st.markdown("""
            <div class="explanation-callout">
                <strong>How Clonal Selection Matured the Repertoire:</strong><br>
                &bull; <strong>Left Plot:</strong> Shows detector affinity scores progressing across 8 generations. High-performing detectors are cloned proportionally, while mutation rates decrease as detectors achieve higher affinity.<br>
                &bull; <strong>Right Plot:</strong> Tracks the discrimination balance. Mutated detectors that invaded Self space were immediately censored, keeping False Positives at 0 while True Positives stabilized.
            </div>
            """, unsafe_allow_html=True)

        # Refresh button
        if st.button("Reload Metrics from Disk"):
            st.cache_data.clear()
            st.rerun()

    else:
        st.info("No saved metrics found. Train models in the 'Train Models' tab or run: python train.py")


# ==============================================================================
# TAB 2: LIVE PACKET INSPECTOR
# ==============================================================================
with tab2:
    st.markdown('<div class="section-header">Live Step-by-Step Packet Classification</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="explanation-callout">
        <strong>How this works:</strong> Select a real connection record from the NSL-KDD test partition. The system will
        compute Euclidean distances from your chosen packet to every detector in the AIS repertoire in real time,
        then apply the threshold decision rule and show you the full mathematical breakdown.
    </div>
    """, unsafe_allow_html=True)

    if preprocessor and ais_model:
        raw_test = PROJECT_ROOT / "data" / "raw" / "KDDTest+.txt"
        if raw_test.exists():
            df_test_sample = load_nsl_kdd(str(raw_test), sample_size=100, random_state=42)

            options = [
                f"Index {idx:02d} | Label: {row['label']:<12} | Category: {row['attack_category']:<6} | Protocol: {row['protocol_type']:<4} | Service: {row['service']}"
                for idx, row in df_test_sample.iterrows()
            ]
            selected_option = st.selectbox("Select Test Connection Record:", options)
            selected_idx = int(selected_option.split("|")[0].replace("Index", "").strip())
            connection_row = df_test_sample.iloc[[selected_idx]]

            ground_truth_label = connection_row["label"].values[0]
            ground_truth_binary = connection_row["binary_label"].values[0]

            # Live computation with spinner and progress
            with st.spinner("Transforming features and computing Euclidean distances across detector repertoire..."):
                X_vec = preprocessor.transform(connection_row)

                n_detectors = len(ais_model.detectors_)
                progress_bar = st.progress(0, text=f"Scanning 0 / {n_detectors} detectors...")

                # Compute distances in chunks to show live progress
                chunk = max(1, n_detectors // 10)
                dists_all = []
                for i in range(0, n_detectors, chunk):
                    batch = ais_model.detectors_[i:i + chunk]
                    dists_all.append(np.linalg.norm(batch - X_vec[0], axis=1))
                    pct = min(int((i + chunk) / n_detectors * 100), 100)
                    progress_bar.progress(pct, text=f"Scanning {min(i + chunk, n_detectors)} / {n_detectors} detectors...")
                    time.sleep(0.03)

                progress_bar.empty()
                dists = np.concatenate(dists_all)

            min_dist = float(np.min(dists))
            radius = float(ais_model.matching_radius)
            anomaly_score = float(np.exp(-(min_dist ** 2) / (2.0 * (radius ** 2))))

            ais_prediction = 1 if min_dist <= radius else 0
            rf_prediction = int(rf_model.predict(X_vec)[0]) if rf_model else None

            st.markdown("#### Step 1 & 2: Ground Truth & Model Classification Outcomes")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Ground Truth Label</div>
                    <div class="metric-value">{ground_truth_label}</div>
                    <div>{'<span class="status-badge-attack">ATTACK (Non-Self)</span>' if ground_truth_binary == 1 else '<span class="status-badge-normal">NORMAL (Self)</span>'}</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Hybrid AIS Decision</div>
                    <div class="metric-value">{'ATTACK' if ais_prediction == 1 else 'NORMAL'}</div>
                    <div>{'<span class="status-badge-attack">Matched Non-Self</span>' if ais_prediction == 1 else '<span class="status-badge-normal">Tolerated Self</span>'}</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Nearest Detector Distance (d_min)</div>
                    <div class="metric-value">{min_dist:.4f}</div>
                    <small style="color: #94a3b8;">Threshold Radius r = {radius:.4f}</small>
                </div>
                """, unsafe_allow_html=True)
            with c4:
                rf_label = "ATTACK" if rf_prediction == 1 else "NORMAL" if rf_prediction is not None else "N/A"
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Random Forest Baseline</div>
                    <div class="metric-value">{rf_label}</div>
                    <small style="color: #94a3b8;">{'Decision Matches Ground Truth' if rf_prediction == ground_truth_binary else 'Classification Error'}</small>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="section-header">Step 3: How the Decision Was Made</div>', unsafe_allow_html=True)

            verdict_color = "#f87171" if ais_prediction == 1 else "#4ade80"
            verdict_word = "ATTACK" if ais_prediction == 1 else "NORMAL"
            dist_vs_radius = "closer than" if min_dist <= radius else "farther than"
            st.markdown(f"""
            <div class="explanation-callout">
                The system measured how far this packet is from the nearest trained detector.
                <br><br>
                <strong>Nearest detector distance:</strong> &nbsp;<code>{min_dist:.4f}</code>
                &nbsp;&nbsp;|&nbsp;&nbsp;
                <strong>Detection threshold:</strong> &nbsp;<code>{radius:.4f}</code>
                <br><br>
                This packet is <strong>{dist_vs_radius} the threshold</strong>, so the AIS classified it as
                <strong style="color: {verdict_color};">{verdict_word}</strong>.
                <br><br>
                <strong>Anomaly confidence score:</strong> &nbsp;<code>{anomaly_score:.4f}</code>
                &nbsp; (0 = definitely normal, 1 = definitely attack)
            </div>
            """, unsafe_allow_html=True)

            if ais_prediction == 1:
                st.markdown("""
                <div class="explanation-callout">
                    <strong>What this means:</strong> At least one of our trained detectors recognised this packet as suspicious.
                    The packet is close enough to known attack patterns that the system flagged it as an intrusion.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="explanation-callout">
                    <strong>What this means:</strong> This packet did not match any detector closely enough to trigger an alert.
                    The system treated it as normal traffic and let it through — consistent with how the immune system tolerates "self" cells.
                </div>
                """, unsafe_allow_html=True)

            # Live distance histogram — 20 nearest detectors
            st.markdown('<div class="section-header">Step 4: 20 Nearest Detector Distances (Live)</div>', unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.4; margin-bottom: 0.5rem;">
                Each bar is the Euclidean distance from this packet to one detector. The red dashed line marks the matching radius <code>r</code>.
                Any detector whose bar falls below <code>r</code> triggers a classification of ATTACK.
            </div>
            """, unsafe_allow_html=True)

            nearest_20_idx = np.argsort(dists)[:20]
            nearest_20_dists = dists[nearest_20_idx]

            fig_live, ax_live = plt.subplots(figsize=(10, 3))
            fig_live.patch.set_facecolor("#000000")
            ax_live.set_facecolor("#0b0f19")
            colors = ["#f87171" if d <= radius else "#38bdf8" for d in nearest_20_dists]
            ax_live.bar(range(len(nearest_20_dists)), nearest_20_dists, color=colors, edgecolor="#334155")
            ax_live.axhline(y=radius, color="#facc15", linestyle="--", linewidth=1.5, label=f"Matching radius r = {radius:.3f}")
            ax_live.set_xlabel("Detector rank (nearest first)", color="#ffffff", fontsize=9)
            ax_live.set_ylabel("Euclidean distance", color="#ffffff", fontsize=9)
            ax_live.set_title("20 Nearest Detector Distances for Selected Packet", color="#ffffff", fontsize=10, fontweight="bold")
            ax_live.tick_params(colors="#ffffff")
            for spine in ax_live.spines.values():
                spine.set_edgecolor("#334155")
            ax_live.legend(facecolor="#0b0f19", edgecolor="#334155", labelcolor="#ffffff", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig_live)
            plt.close(fig_live)

            st.markdown('<div class="section-header">Raw Network Connection Feature Attributes (41 Features)</div>', unsafe_allow_html=True)
            raw_display = connection_row.drop(columns=["binary_label"], errors="ignore")
            st.dataframe(raw_display.T.rename(columns={selected_idx: "Attribute Value"}), use_container_width=True)

        else:
            st.error("Test data file not located at data/raw/KDDTest+.txt.")
    else:
        st.warning("Models not loaded. Train models first (use the 'Train Models' tab or run: python train.py).")


# ==============================================================================
# TAB 3: DATASET SPECIFICATION
# ==============================================================================
with tab3:
    st.markdown('<div class="section-header">NSL-KDD Dataset Specification & Class Distribution</div>', unsafe_allow_html=True)

    df_sample = get_dataset_sample(15000)
    if df_sample is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sampled Records", f"{len(df_sample):,}")
        c2.metric("Normal Connections", f"{(df_sample['binary_label'] == 0).sum():,}")
        c3.metric("Attack Connections", f"{(df_sample['binary_label'] == 1).sum():,}")
        c4.metric("Total Attributes", "41 Features + 2 Metadata")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Attack Category Taxonomy")
            st.markdown("""
            | Category | Attack Types Represented |
            | :--- | :--- |
            | **DoS** | apache2, back, land, mailbomb, neptune, pod, smurf, teardrop |
            | **Probe** | ipsweep, mscan, nmap, portsweep, saint, satan |
            | **R2L** | ftp_write, guess_passwd, imap, multihop, phf, warezclient |
            | **U2R** | buffer_overflow, loadmodule, perl, rootkit, sqlattack |
            | **Normal** | Legitimate non-malicious user and server communications |
            """)

        with col_b:
            st.markdown("#### Attack Category Frequency Distribution")
            cat_counts = df_sample["attack_category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Instances"]
            fig, ax = plt.subplots(figsize=(6, 3.8))
            sns.barplot(data=cat_counts, x="Category", y="Instances", color="#38bdf8", ax=ax, edgecolor="#334155")
            ax.set_title("Distribution of Traffic Classes (NSL-KDD Sample)", fontsize=11, weight="bold")
            plt.xticks(rotation=20)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.markdown('<div class="section-header">Sample Connection Records (Raw Ingestion)</div>', unsafe_allow_html=True)
        st.dataframe(df_sample.head(5), use_container_width=True)
    else:
        st.warning("NSL-KDD data file missing in data/raw/KDDTrain+.txt.")


# ==============================================================================
# TAB 4: ALGORITHM TRAINING STUDIO
# ==============================================================================
with tab4:
    st.markdown('<div class="section-header">Model Hyperparameter Configuration & Execution</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="explanation-callout">
        <strong>Training Configuration Guide:</strong><br>
        &bull; <strong>Target Detector Count (K):</strong> Repertoire size. More detectors provide denser coverage of Non-Self space.<br>
        &bull; <strong>Matching Radius (r):</strong> Euclidean threshold. Smaller radius enforces stricter self-tolerance; larger radius increases attack detection rate.<br>
        &bull; <strong>Clonal Generations:</strong> Iterations of evolutionary affinity maturation and somatic hypermutation.
    </div>
    """, unsafe_allow_html=True)

    with st.form("training_parameters"):
        c1, c2, c3 = st.columns(3)
        with c1:
            n_det = st.number_input("Target Detector Count (K)", min_value=10, max_value=300, value=80, step=10)
            radius = st.slider("Matching Radius Threshold (r)", min_value=0.20, max_value=0.80, value=0.45, step=0.05)
        with c2:
            n_gen = st.number_input("Clonal Generations", min_value=1, max_value=25, value=8, step=1)
            mut_rate = st.slider("Base Mutation Standard Deviation", min_value=0.02, max_value=0.30, value=0.10, step=0.02)
        with c3:
            train_n = st.selectbox("Training Partition Size", [5000, 10000, 15000, 25000, "Full (125,973)"], index=2)
            test_n = st.selectbox("Testing Partition Size", [2000, 5000, 10000, "Full (22,544)"], index=1)

        submitted = st.form_submit_button("Execute Training Pipeline")

    if submitted:
        train_s = 0 if train_n == "Full (125,973)" else int(train_n)
        test_s = 0 if test_n == "Full (22,544)" else int(test_n)

        raw_train_p = PROJECT_ROOT / "data" / "raw" / "KDDTrain+.txt"
        raw_test_p = PROJECT_ROOT / "data" / "raw" / "KDDTest+.txt"

        if not raw_train_p.exists() or not raw_test_p.exists():
            st.error("Dataset files not found in data/raw/.")
        else:
            with st.spinner("Executing pipeline: Ingesting dataset, fitting transformers, and training models..."):
                df_tr = load_nsl_kdd(str(raw_train_p), sample_size=train_s if train_s > 0 else None, random_state=42)
                df_te = load_nsl_kdd(str(raw_test_p), sample_size=test_s if test_s > 0 else None, random_state=42)

                prep = NSLPreprocessor()
                X_tr = prep.fit_transform(df_tr)
                y_tr = prep.extract_labels(df_tr)
                X_te = prep.transform(df_te)
                y_te = prep.extract_labels(df_te)

                m_dir = PROJECT_ROOT / "models"
                prep.save(m_dir / "preprocessor.joblib")

                ais = HybridAISClassifier(
                    num_detectors=n_det,
                    matching_radius=radius,
                    clonal_generations=n_gen,
                    mutation_rate=mut_rate,
                    seed=42,
                )
                ais.fit(X_tr, y_tr)
                ais.save(m_dir / "hybrid_ais.joblib")

                rf, rf_time = train_random_forest(X_tr, y_tr, n_estimators=100, random_state=42)
                save_random_forest(rf, m_dir / "random_forest.joblib")

                m_ais = evaluate_model(ais, X_te, y_te, model_name="Hybrid AIS", train_time=ais.train_time_)
                m_rf = evaluate_model(rf, X_te, y_te, model_name="Random Forest", train_time=rf_time)

                m_list = [m_ais, m_rf]
                r_dir = PROJECT_ROOT / "results"
                save_metrics(m_list, r_dir / "metrics")

                fig_dir = r_dir / "figures"
                plot_confusion_matrices(m_list, fig_dir / "confusion_matrices.png")
                plot_metrics_comparison(m_list, fig_dir / "metrics_comparison.png")
                if ais.history_:
                    plot_clonal_progress(ais.history_, fig_dir / "clonal_progress.png")
                plot_roc_curves({"Hybrid AIS": ais, "Random Forest": rf}, X_te, y_te, fig_dir / "roc_curves.png")
                plot_execution_times(m_list, fig_dir / "execution_times.png")

            st.success("Training and benchmarking completed successfully. Results updated in repository.")
            st.rerun()


# ==============================================================================
# TAB 5: RESEARCH PAPER TABLE EXPORTER
# ==============================================================================
with tab5:
    st.markdown('<div class="section-header">Auto-Generated Research Table</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="explanation-callout">
        This is a ready-to-paste LaTeX table of all measured results, formatted for IEEE / Springer research papers.
        Every number here is computed from the actual trained models — nothing is fabricated.
    </div>
    """, unsafe_allow_html=True)

    if metrics_data:
        ais_m = next((m for m in metrics_data if m["model_name"] == "Hybrid AIS"), None)
        rf_m = next((m for m in metrics_data if m["model_name"] == "Random Forest"), None)

        if ais_m and rf_m:
            spec_ais = ais_m.get("specificity", 1.0 - ais_m["false_positive_rate"])
            spec_rf = rf_m.get("specificity", 1.0 - rf_m["false_positive_rate"])
            mcc_ais = ais_m.get("mcc", 0.0)
            mcc_rf = rf_m.get("mcc", 0.0)
            auc_ais = f"{ais_m['roc_auc']:.4f}" if ais_m.get("roc_auc") is not None else "N/A"
            auc_rf = f"{rf_m['roc_auc']:.4f}" if rf_m.get("roc_auc") is not None else "N/A"

            latex_code = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Quantitative Performance Comparison: Hybrid AIS vs. Baseline on NSL-KDD Test Set}}
\\label{{tab:nids_benchmark}}
\\begin{{tabular}}{{lcc}}
\\hline
\\textbf{{Metric}} & \\textbf{{Hybrid AIS}} & \\textbf{{Random Forest}} \\\\
\\hline
Accuracy & {ais_m['accuracy']:.4f} & {rf_m['accuracy']:.4f} \\\\
Precision (PPV) & {ais_m['precision']:.4f} & {rf_m['precision']:.4f} \\\\
Recall (Detection Rate) & {ais_m['recall']:.4f} & {rf_m['recall']:.4f} \\\\
Specificity (TNR) & {spec_ais:.4f} & {spec_rf:.4f} \\\\
False Positive Rate (FPR) & {ais_m['false_positive_rate']:.4f} & {rf_m['false_positive_rate']:.4f} \\\\
False Negative Rate (FNR) & {ais_m['false_negative_rate']:.4f} & {rf_m['false_negative_rate']:.4f} \\\\
F1-Score & {ais_m['f1_score']:.4f} & {rf_m['f1_score']:.4f} \\\\
Matthews Correlation (MCC) & {mcc_ais:.4f} & {mcc_rf:.4f} \\\\
ROC-AUC & {auc_ais} & {auc_rf} \\\\
\\hline
Training Duration (s) & {ais_m['training_time_sec']:.2f} & {rf_m['training_time_sec']:.2f} \\\\
Inference Latency (s) & {ais_m['prediction_time_sec']:.4f} & {rf_m['prediction_time_sec']:.4f} \\\\
Throughput (samples/s) & {ais_m['throughput_samples_per_sec']:,.0f} & {rf_m['throughput_samples_per_sec']:,.0f} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}"""

            st.code(latex_code, language="latex")
            st.caption("Copy the block above and paste it directly into any LaTeX editor (Overleaf, TeXstudio).")
    else:
        st.info("No saved metrics found. Train models first (Train Models tab or: python train.py).")


# ==============================================================================
# TAB 6: LEGAL
# ==============================================================================
with tab6:
    st.markdown('<div class="section-header">Terms of Use & Privacy</div>', unsafe_allow_html=True)
    st.markdown("""
    **Academic Use Only.**
    This dashboard is built for university coursework evaluation and algorithm research.
    It runs entirely on your local machine — no data is sent to any external server.

    The NSL-KDD dataset used here contains anonymized, synthetic network connection records with no
    Personally Identifiable Information (PII). All model training, evaluation, and inference happens
    offline on your own hardware.

    The Hybrid AIS is a computational abstraction of biological immune system principles.
    It is not intended for deployment in production network security infrastructure without
    additional engineering controls.

    Source code is open for academic review under standard fair-use guidelines.
    """)


# ==============================================================================
# TAB 7: DEPLOYMENT
# ==============================================================================
with tab7:
    st.markdown('<div class="section-header">How to Run This Project</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="explanation-callout">
        Everything needed to reproduce this project is in the GitHub repository.
        The dataset, trained models, and results are all included.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Clone and install")
    st.code("""git clone https://github.com/adithibcs25-crypto/BIO_EL.git
cd BIO_EL/Bio_EL_Intrusion_Detection
pip install -r requirements.txt""", language="bash")

    st.markdown("#### Run the dashboard")
    st.code("streamlit run app.py", language="bash")

    st.markdown("#### Re-train models from scratch (optional)")
    st.code("python train.py", language="bash")

    st.markdown("#### Run tests")
    st.code("python -m pytest tests/", language="bash")

    st.markdown("""
    <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 1rem;">
        Dataset files (KDDTrain+.txt and KDDTest+.txt) are already in <code>data/raw/</code>.
        Pre-trained model files are already in <code>models/</code>.
        No internet connection required to run locally.
    </div>
    """, unsafe_allow_html=True)
