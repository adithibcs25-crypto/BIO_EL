"""Data loading and verification module for the NSL-KDD dataset."""

import os
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd


COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty"
]

CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]

NUMERICAL_COLUMNS = [
    col for col in COLUMN_NAMES if col not in CATEGORICAL_COLUMNS + ["label", "difficulty"]
]

# Attack categories mapping for detailed breakdown
ATTACK_CATEGORIES = {
    "dos": [
        "apache2", "back", "land", "mailbomb", "neptune",
        "pod", "processtable", "smurf", "teardrop", "udpstorm"
    ],
    "probe": [
        "ipsweep", "mscan", "nmap", "portsweep", "saint", "satan"
    ],
    "r2l": [
        "ftp_write", "guess_passwd", "httptunnel", "imap", "multihop",
        "named", "phf", "sendmail", "snmpgetattack", "snmpguess",
        "spy", "warezclient", "warezmaster", "worm", "xlock", "xsnoop"
    ],
    "u2r": [
        "buffer_overflow", "loadmodule", "perl", "ps", "rootkit",
        "sqlattack", "xterm"
    ]
}


def get_attack_category(attack_name: str) -> str:
    """Map a specific NSL-KDD attack name to its high-level attack category."""
    label_clean = str(attack_name).strip().lower()
    if label_clean == "normal":
        return "normal"
    for category, attacks in ATTACK_CATEGORIES.items():
        if label_clean in attacks:
            return category
    return "other_attack"


def load_nsl_kdd(
    file_path: str,
    nrows: Optional[int] = None,
    sample_size: Optional[int] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """Load an NSL-KDD dataset file into a structured pandas DataFrame.

    Parameters
    ----------
    file_path : str
        Path to the KDDTrain+.txt or KDDTest+.txt file.
    nrows : int, optional
        Number of rows to read from the start of the file.
    sample_size : int, optional
        If specified, randomly sample this number of rows (reproducible via random_state).
    random_state : int
        Seed for sampling.

    Returns
    -------
    pd.DataFrame
        DataFrame with 43 columns named according to the NSL-KDD specification,
        plus an added 'binary_label' column (0 for normal, 1 for attack).
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"NSL-KDD dataset file not found at: {path.resolve()}\n"
            "Please ensure 'KDDTrain+.txt' and 'KDDTest+.txt' are placed inside "
            "the 'data/raw/' directory. Refer to README.md for download instructions."
        )

    df = pd.read_csv(path, header=None, names=COLUMN_NAMES, nrows=nrows)

    # Clean label string
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    # Binary label: 0 for normal, 1 for attack
    df["binary_label"] = (df["label"] != "normal").astype(int)

    # High-level category
    df["attack_category"] = df["label"].apply(get_attack_category)

    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    return df


def load_train_test(
    data_dir: str,
    train_filename: str = "KDDTrain+.txt",
    test_filename: str = "KDDTest+.txt",
    train_sample_size: Optional[int] = None,
    test_sample_size: Optional[int] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load both training and test NSL-KDD datasets."""
    train_path = os.path.join(data_dir, train_filename)
    test_path = os.path.join(data_dir, test_filename)

    df_train = load_nsl_kdd(
        train_path,
        sample_size=train_sample_size,
        random_state=random_state
    )
    df_test = load_nsl_kdd(
        test_path,
        sample_size=test_sample_size,
        random_state=random_state
    )

    return df_train, df_test
