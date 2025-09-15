# src/ml/preprocess.py
"""
Production-ready preprocessing utilities for ticket classification.

Functions:
- load_raw: loads raw CSV
- clean_text: basic cleaning
- prepare_dataset: filter, label-map, split, save artifacts, return train/test sets
"""

from pathlib import Path
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from typing import Optional, Tuple, Dict, Any

ROOT = Path(__file__).resolve().parents[2]   # project root
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Regex patterns
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
NON_ALPHANUM_RE = re.compile(r"[^a-z0-9\s]")

def load_raw(filename: str) -> pd.DataFrame:
    """Load raw CSV from data/raw."""
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return pd.read_csv(path)

def clean_text(series: pd.Series) -> pd.Series:
    """Clean text for modeling."""
    s = series.fillna("").astype(str).str.lower()
    s = s.str.replace(URL_RE, "", regex=True)
    s = s.str.replace(MENTION_RE, "", regex=True)
    s = s.str.replace(r"rt\s?:", "", regex=True)
    s = s.str.replace(NON_ALPHANUM_RE, " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

def prepare_dataset(
    filename: str,
    text_columns: Tuple[str, ...] = ("subject", "body"),
    label_col: str = "queue",
    language: Optional[str] = None,
    min_samples_per_label: int = 50,
    test_size: float = 0.2,
    random_state: int = 42,
    save_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Preprocess dataset and return train/test sets along with summary.

    Returns:
        dict: {
            'X_train', 'X_test', 'y_train', 'y_test',
            'train_path', 'test_path', 'label_encoder_path',
            'summary'
        }
    """
    df = load_raw(filename)

    # Keep only customer inbound messages if column exists
    if "inbound" in df.columns:
        df = df[df["inbound"] == True]

    # Combine text columns
    texts = []
    for c in text_columns:
        if c in df.columns:
            texts.append(df[c].astype(str))
    if not texts:  # fallback
        for c in ["text", "body", "subject"]:
            if c in df.columns:
                texts = [df[c].astype(str)]
                break
    if not texts:
        raise ValueError("No text columns found in dataset!")

    # Join multiple series into single text column
    df["text"] = pd.Series([" ".join(row) if isinstance(row, list) else row for row in zip(*texts)])

    # Filter by language if needed
    if language and "language" in df.columns:
        df = df[df["language"] == language]

    # Drop missing text or label
    df = df.dropna(subset=["text", label_col]).reset_index(drop=True)

    # Clean text
    df["text_clean"] = clean_text(df["text"])

    # Extract top-level label (split by / or :)
    df["queue_top"] = df[label_col].astype(str).apply(lambda s: s.split("/")[0].split(":")[0].strip())

    # Map rare labels to 'Other'
    counts = df["queue_top"].value_counts()
    rare = counts[counts < min_samples_per_label].index.tolist()
    df["label_final"] = df["queue_top"].apply(lambda x: "Other" if x in rare else x)

    # Label encode
    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["label_final"])

    # Train/test split
    X = df["text_clean"]
    y = df["label_id"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Save paths
    save_dir = save_dir or PROCESSED_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    train_path = save_dir / "train.csv"
    test_path = save_dir / "test.csv"
    label_encoder_path = save_dir / "label_encoder.joblib"

    # Save CSVs
    pd.DataFrame({"text": X_train, "label": y_train}).to_csv(train_path, index=False)
    pd.DataFrame({"text": X_test, "label": y_test}).to_csv(test_path, index=False)
    joblib.dump(le, label_encoder_path)

    summary = {
        "n_rows_raw": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "label_classes": list(le.classes_),
        "label_counts": df["label_final"].value_counts().to_dict(),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "label_encoder": str(label_encoder_path),
    }

    return {
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "train_path": str(train_path),
        "test_path": str(test_path),
        "label_encoder_path": str(label_encoder_path),
        "summary": summary
    }
