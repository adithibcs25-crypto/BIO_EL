"""End-to-end training and benchmarking pipeline for Bio_EL_Intrusion_Detection.

Trains both the Hybrid Artificial Immune System and baseline Random Forest,
evaluates on unseen test traffic, and generates persistent artifacts.
"""

import argparse
import os
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set matplotlib cache directory to local results to avoid Windows permissions issues
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "results" / ".mpl_cache"))

from src.data_loader import load_nsl_kdd
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


def run_pipeline(
    data_dir: str = "data/raw",
    train_file: str = "KDDTrain+.txt",
    test_file: str = "KDDTest+.txt",
    train_sample_size: int = 15000,
    test_sample_size: int = 5000,
    num_detectors: int = 80,
    matching_radius: float = 0.45,
    clonal_generations: int = 8,
    mutation_rate: float = 0.1,
    clone_factor: float = 4.0,
    random_state: int = 42,
    models_dir: str = "models",
    results_dir: str = "results",
) -> None:
    """Execute complete end-to-end training and evaluation pipeline."""
    print("=" * 75)
    print(" AI-Based Network Intrusion Detection: Hybrid Artificial Immune System")
    print("=" * 75)

    base_dir = Path(PROJECT_ROOT)
    data_path = base_dir / data_dir
    train_path = data_path / train_file
    test_path = data_path / test_file

    # 1. Dataset Loading
    print(f"\n[1/6] Loading NSL-KDD dataset from: {data_path.resolve()} ...")
    if not train_path.exists() or not test_path.exists():
        print(f"ERROR: Dataset files not found in {data_path.resolve()}")
        print("Please check that KDDTrain+.txt and KDDTest+.txt exist.")
        sys.exit(1)

    df_train = load_nsl_kdd(
        str(train_path),
        sample_size=train_sample_size if train_sample_size > 0 else None,
        random_state=random_state,
    )
    df_test = load_nsl_kdd(
        str(test_path),
        sample_size=test_sample_size if test_sample_size > 0 else None,
        random_state=random_state,
    )

    print(f"      Training set: {len(df_train):,} samples | Attack ratio: {df_train['binary_label'].mean():.1%}")
    print(f"      Testing set:  {len(df_test):,} samples | Attack ratio: {df_test['binary_label'].mean():.1%}")

    # 2. Preprocessing
    print("\n[2/6] Preprocessing data (MinMax scaling + OneHot encoding) ...")
    preprocessor = NSLPreprocessor()
    X_train = preprocessor.fit_transform(df_train)
    y_train = preprocessor.extract_labels(df_train)

    # Strictly transform test data using fitted transformers (NO DATA LEAKAGE)
    X_test = preprocessor.transform(df_test)
    y_test = preprocessor.extract_labels(df_test)

    # Save preprocessor artifact
    m_dir = base_dir / models_dir
    m_dir.mkdir(parents=True, exist_ok=True)
    prep_path = m_dir / "preprocessor.joblib"
    preprocessor.save(prep_path)
    print(f"      Feature dimensionality: {X_train.shape[1]} features")
    print(f"      Saved preprocessor -> {prep_path}")

    # 3. Hybrid AIS Training
    print(f"\n[3/6] Training Hybrid AIS Classifier ...")
    print(f"      Parameters: detectors={num_detectors}, radius={matching_radius}, "
          f"generations={clonal_generations}, mutation_rate={mutation_rate}")
    ais = HybridAISClassifier(
        num_detectors=num_detectors,
        matching_radius=matching_radius,
        clonal_generations=clonal_generations,
        clone_factor=clone_factor,
        mutation_rate=mutation_rate,
        seed=random_state,
    )
    ais.fit(X_train, y_train)
    ais_path = m_dir / "hybrid_ais.joblib"
    ais.save(ais_path)
    print(f"      Hybrid AIS training completed in {ais.train_time_:.2f}s")
    print(f"      Saved AIS model -> {ais_path}")

    # 4. Baseline Random Forest Training
    print(f"\n[4/6] Training Baseline Random Forest Classifier ...")
    rf, rf_train_time = train_random_forest(
        X_train,
        y_train,
        n_estimators=100,
        random_state=random_state,
    )
    rf_path = m_dir / "random_forest.joblib"
    save_random_forest(rf, rf_path)
    print(f"      Random Forest training completed in {rf_train_time:.2f}s")
    print(f"      Saved RF model -> {rf_path}")

    # 5. Model Evaluation
    print(f"\n[5/6] Evaluating models on unseen test partition ({len(X_test):,} samples) ...")
    metrics_ais = evaluate_model(ais, X_test, y_test, model_name="Hybrid AIS", train_time=ais.train_time_)
    metrics_rf = evaluate_model(rf, X_test, y_test, model_name="Random Forest", train_time=rf_train_time)

    metrics_list = [metrics_ais, metrics_rf]
    r_dir = base_dir / results_dir
    m_metrics_dir = r_dir / "metrics"
    df_results = save_metrics(metrics_list, m_metrics_dir)
    print(f"      Saved metrics to -> {m_metrics_dir}")

    # 6. Visualization Generation
    print(f"\n[6/6] Generating benchmark figures and plots ...")
    fig_dir = r_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrices(metrics_list, fig_dir / "confusion_matrices.png")
    plot_metrics_comparison(metrics_list, fig_dir / "metrics_comparison.png")
    if ais.history_:
        plot_clonal_progress(ais.history_, fig_dir / "clonal_progress.png")
    plot_roc_curves({"Hybrid AIS": ais, "Random Forest": rf}, X_test, y_test, fig_dir / "roc_curves.png")
    plot_execution_times(metrics_list, fig_dir / "execution_times.png")
    print(f"      Saved figures to -> {fig_dir}")

    # Console Summary Table
    print("\n" + "=" * 75)
    print(" BENCHMARK RESULTS SUMMARY (NSL-KDD Test Set)")
    print("=" * 75)
    header = f"{'Metric':<25} | {'Hybrid AIS':<18} | {'Random Forest':<18}"
    print(header)
    print("-" * len(header))
    print(f"{'Accuracy':<25} | {metrics_ais['accuracy']:<18.4f} | {metrics_rf['accuracy']:<18.4f}")
    print(f"{'Precision (PPV)':<25} | {metrics_ais['precision']:<18.4f} | {metrics_rf['precision']:<18.4f}")
    print(f"{'Recall (Detection Rate)':<25} | {metrics_ais['recall']:<18.4f} | {metrics_rf['recall']:<18.4f}")
    print(f"{'Specificity (TNR)':<25} | {metrics_ais['specificity']:<18.4f} | {metrics_rf['specificity']:<18.4f}")
    print(f"{'F1-Score':<25} | {metrics_ais['f1_score']:<18.4f} | {metrics_rf['f1_score']:<18.4f}")
    print(f"{'False Positive Rate (FPR)':<25} | {metrics_ais['false_positive_rate']:<18.4f} | {metrics_rf['false_positive_rate']:<18.4f}")
    print(f"{'Matthews Corr (MCC)':<25} | {metrics_ais['mcc']:<18.4f} | {metrics_rf['mcc']:<18.4f}")
    auc_ais = f"{metrics_ais['roc_auc']:.4f}" if metrics_ais['roc_auc'] else "N/A"
    auc_rf = f"{metrics_rf['roc_auc']:.4f}" if metrics_rf['roc_auc'] else "N/A"
    print(f"{'ROC AUC':<25} | {auc_ais:<18} | {auc_rf:<18}")
    print(f"{'Training Time (s)':<25} | {metrics_ais['training_time_sec']:<18.2f} | {metrics_rf['training_time_sec']:<18.2f}")
    print(f"{'Prediction Latency (s)':<25} | {metrics_ais['prediction_time_sec']:<18.2f} | {metrics_rf['prediction_time_sec']:<18.2f}")
    print(f"{'Throughput (samples/s)':<25} | {metrics_ais['throughput_samples_per_sec']:<18.1f} | {metrics_rf['throughput_samples_per_sec']:<18.1f}")
    print("=" * 75)
    print("Pipeline execution completed successfully!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hybrid AIS and RF on NSL-KDD.")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Path to raw data directory")
    parser.add_argument("--train_file", type=str, default="KDDTrain+.txt", help="Train filename")
    parser.add_argument("--test_file", type=str, default="KDDTest+.txt", help="Test filename")
    parser.add_argument("--train_sample", type=int, default=15000, help="Train sample count (0 for full)")
    parser.add_argument("--test_sample", type=int, default=5000, help="Test sample count (0 for full)")
    parser.add_argument("--detectors", type=int, default=80, help="Number of AIS detectors")
    parser.add_argument("--radius", type=float, default=0.45, help="AIS matching radius")
    parser.add_argument("--generations", type=int, default=8, help="Clonal selection generations")
    parser.add_argument("--mutation_rate", type=float, default=0.1, help="AIS mutation rate")
    parser.add_argument("--clone_factor", type=float, default=4.0, help="AIS clone factor")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save results")

    args = parser.parse_args()
    run_pipeline(
        data_dir=args.data_dir,
        train_file=args.train_file,
        test_file=args.test_file,
        train_sample_size=args.train_sample,
        test_sample_size=args.test_sample,
        num_detectors=args.detectors,
        matching_radius=args.radius,
        clonal_generations=args.generations,
        mutation_rate=args.mutation_rate,
        clone_factor=args.clone_factor,
        random_state=args.seed,
        models_dir=args.models_dir,
        results_dir=args.results_dir,
    )
