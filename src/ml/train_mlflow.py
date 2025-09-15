"""
Train multiple models for ticket classification and log everything to MLflow.

- Loads dataset using `preprocess.py`.
- Trains multiple classifiers.
- Logs parameters, metrics, and artifacts to MLflow.
- Registers the best model in MLflow Registry.
"""

import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from src.ml.preprocess import prepare_dataset
import seaborn as sns
import matplotlib.pyplot as plt
import json
import warnings

warnings.filterwarnings("ignore")

# ===== CONFIG =====
DATA_FILE = "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
LABEL_COL = "queue"
TEXT_COLUMNS = ("body", "subject")
MIN_SAMPLES_PER_LABEL = 50
TEST_SIZE = 0.2
RANDOM_STATE = 42

MLFLOW_EXPERIMENT = "ticket_classification"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"  # adjust if remote
MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "LinearSVC": LinearSVC(max_iter=10000, random_state=RANDOM_STATE)
}

# ===== STEP 1: PREPROCESS DATA =====
data = prepare_dataset(
    filename=DATA_FILE,
    text_columns=TEXT_COLUMNS,
    label_col=LABEL_COL,
    language=None,
    min_samples_per_label=MIN_SAMPLES_PER_LABEL,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]

# Load Label Encoder for later use
le = data["label_encoder_path"]

print("Dataset prepared:")
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# ===== STEP 2: SET MLFLOW =====
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

best_model_name = None
best_metric = 0.0  # track best macro-F1

# ===== STEP 3: TRAIN & LOG MODELS =====
for model_name, model in MODELS.items():
    with mlflow.start_run(run_name=model_name):
        print(f"\nTraining {model_name}...")

        # Pipeline: TF-IDF + Classifier
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
            ("clf", model)
        ])

        # Train
        pipeline.fit(X_train, y_train)

        # Predict
        y_pred = pipeline.predict(X_test)

        # ===== METRICS =====
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")
        print(f"{model_name} Accuracy: {acc:.4f}, Macro-F1: {f1_macro:.4f}, Weighted-F1: {f1_weighted:.4f}")

        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_dir = Path("reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{model_name}_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        # Confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=data["summary"]["label_classes"], 
                    yticklabels=data["summary"]["label_classes"], cmap="Blues")
        plt.title(f"{model_name} Confusion Matrix")
        cm_path = report_dir / f"{model_name}_confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()

        # ===== LOG TO MLFLOW =====
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("tfidf_max_features", 20000)
        mlflow.log_param("tfidf_ngram_range", (1,2))
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.log_metric("f1_weighted", f1_weighted)
        mlflow.log_artifact(report_path)
        mlflow.log_artifact(cm_path)
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=model_name
        )

        # Track best model (macro-F1)
        if f1_macro > best_metric:
            best_metric = f1_macro
            best_model_name = model_name

print(f"\nBest model: {best_model_name} with Macro-F1 {best_metric:.4f}")
