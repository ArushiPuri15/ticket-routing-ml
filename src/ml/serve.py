# src/ml/serve.py
"""
Model serving utilities for ticket classification.

- Loads model from MLflow Registry (prefer Production stage; fall back to latest).
- Loads label encoder saved in data/processed/label_encoder.joblib.
- Exposes TicketClassifier.predict_texts(texts: List[str]) -> List[dict].
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import joblib
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.pyfunc
import numpy as np
import math

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

# Config - adjust if needed
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_MODEL_NAME = "LinearSVC"     # change to your registered model family name if different
LABEL_ENCODER_PATH = Path("data/processed/label_encoder.joblib")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
_client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


class TicketClassifier:
    _pyfunc_model = None
    _label_encoder = None
    _model_name = DEFAULT_MODEL_NAME

    @classmethod
    def _resolve_model_uri(cls, model_name: str) -> str:
        """
        Resolve model URI to load. Prefer stage 'Production' if available;
        otherwise use the latest registered version.
        """
        try:
            # check for production versions
            prod_versions = _client.get_latest_versions(name=model_name, stages=["Production"])
            if prod_versions:
                v = prod_versions[0].version
                uri = f"models:/{model_name}/Production"
                LOG.info("Found Production model for %s version=%s -> %s", model_name, v, uri)
                return uri

            # fallback: latest version (un- staged)
            # We'll fetch highest version number
            all_versions = _client.get_latest_versions(name=model_name)
            if all_versions:
                # get the latest (highest) version regardless of stage
                versions = sorted(all_versions, key=lambda vv: int(vv.version), reverse=True)
                chosen = versions[0].version
                uri = f"models:/{model_name}/{chosen}"
                LOG.info("No Production model; falling back to latest version %s -> %s", chosen, uri)
                return uri

        except Exception as e:
            LOG.warning("Error resolving model from registry: %s", e)

        # final fallback to "models:/<name>/latest" — pyfunc loader will try
        uri = f"models:/{model_name}/latest"
        LOG.info("Falling back to model URI: %s", uri)
        return uri

    @classmethod
    def load_model(cls, model_name: Optional[str] = None):
        """Lazy-load the pyfunc model and label encoder."""
        if model_name:
            cls._model_name = model_name

        if cls._pyfunc_model is None:
            model_uri = cls._resolve_model_uri(cls._model_name)
            LOG.info("Loading model from MLflow URI: %s", model_uri)
            cls._pyfunc_model = mlflow.pyfunc.load_model(model_uri)
            LOG.info("Model loaded.")

        if cls._label_encoder is None:
            if LABEL_ENCODER_PATH.exists():
                cls._label_encoder = joblib.load(LABEL_ENCODER_PATH)
                LOG.info("Label encoder loaded from %s", LABEL_ENCODER_PATH)
            else:
                LOG.warning("Label encoder not found at %s - predictions may be numeric ids", LABEL_ENCODER_PATH)
                cls._label_encoder = None

        return cls._pyfunc_model, cls._label_encoder

    @staticmethod
    def _safe_predict_proba(model, texts: List[str]) -> Optional[List[float]]:
        """Return max predicted probability per sample if available; else None."""
        try:
            # For sklearn pipelines, predict_proba may be supported if classifier supports it
            probs = model.predict_proba(texts)  # may raise
            # probs shape: (n_samples, n_classes)
            max_probs = probs.max(axis=1).tolist()
            return [float(p) for p in max_probs]
        except Exception:
            # try decision_function (gives scores) and convert to softmax
            try:
                scores = model.decision_function(texts)  # shape (n_samples, n_classes) or (n_samples,)
                if isinstance(scores, list) or (hasattr(scores, "shape") and scores.ndim == 2):
                    # apply softmax rowwise
                    def softmax(arr):
                        exps = np.exp(arr - np.max(arr))
                        return exps / exps.sum()
                    out = []
                    for row in scores:
                        sm = softmax(row)
                        out.append(float(np.max(sm)))
                    return out
                else:
                    # binary case: decision_function returns single score -> convert via sigmoid
                    def sigmoid(x): return 1.0 / (1.0 + math.exp(-x))
                    return [float(sigmoid(s)) for s in scores]
            except Exception:
                return None

    @classmethod
    def predict_texts(cls, texts: List[str], model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Predict labels for a batch of texts.

        Returns a list of dicts:
            {
              "label": <str or None>,
              "label_id": <int or None>,
              "confidence": <float or null>
            }
        """
        model, le = cls.load_model(model_name)
        # pyfunc expects a pandas Series or list; pass list
        raw_preds = model.predict(texts)

        # convert to numpy array for consistency
        preds_arr = np.asarray(raw_preds)
        # try to convert to ints (label ids)
        try:
            label_ids = preds_arr.astype(int).tolist()
        except Exception:
            # maybe model returned strings (if you trained with string labels)
            label_ids = None

        # map ids -> human labels if label encoder available
        results = []
        confidences = cls._safe_predict_proba(model, texts)

        for i, raw in enumerate(raw_preds):
            item: Dict[str, Any] = {}
            if le is not None and label_ids is not None:
                # inverse transform
                try:
                    label = le.inverse_transform([label_ids[i]])[0]
                    item["label"] = str(label)
                    item["label_id"] = int(label_ids[i])
                except Exception:
                    # if inverse fails
                    item["label"] = str(raw)
                    item["label_id"] = int(label_ids[i]) if label_ids is not None else None
            else:
                # no label encoder; return raw
                item["label"] = str(raw)
                item["label_id"] = int(raw) if (isinstance(raw, (int, np.integer)) or (str(raw).isdigit())) else None

            item["confidence"] = float(confidences[i]) if (confidences is not None and confidences[i] is not None) else None
            results.append(item)

        return results
