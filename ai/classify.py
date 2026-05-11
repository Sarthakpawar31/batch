"""
ai/classify.py

TensorFlow Lite disease classification engine.

Model: MobileNetV2 (224×224, float32 inputs, softmax outputs)
       fine-tuned on PlantVillage dataset, exported as .tflite.

Memory budget:
  • MobileNetV2 .tflite  ≈ 14 MB on disk
  • Loaded interpreter   ≈ 30 MB RSS
  • Input tensor buffer  ≈ 0.6 MB  (224×224×3 float32)

Total: well within the 1 GB limit when combined with the rest of
the system (target ≤ 250 MB overall).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from config import AI_CFG, MODEL_PATH

logger = logging.getLogger(__name__)

# ── TFLite import with fallback stub ─────────────────────────────────────────
try:
    import tflite_runtime.interpreter as tflite   # type: ignore[import]
    _TFLITE_OK = True
except ImportError:
    try:
        import tensorflow.lite as tflite          # type: ignore[import]
        _TFLITE_OK = True
    except ImportError:
        logger.warning("tflite_runtime not found – classifier in STUB mode.")
        _TFLITE_OK = False


@dataclass
class PredictionResult:
    disease:    str
    confidence: float          # 0.0 – 1.0
    severity:   str            # "none" | "low" | "moderate" | "high"
    top_k:      List[tuple]    # [(label, score), ...]
    is_healthy: bool


def _severity_from_confidence(conf: float) -> str:
    if conf < 0.70:
        return "low"
    if conf < 0.85:
        return "moderate"
    return "high"


def _preprocess(image: np.ndarray) -> np.ndarray:
    """
    Resize to 224×224, convert BGR→RGB, normalise to [-1, 1]
    (MobileNetV2 convention), add batch dimension.
    Returns float32 array shape (1, 224, 224, 3).
    """
    resized = cv2.resize(image, (AI_CFG.INPUT_SIZE, AI_CFG.INPUT_SIZE),
                         interpolation=cv2.INTER_LINEAR)
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    arr     = rgb.astype(np.float32)
    arr     = (arr / 127.5) - 1.0        # [-1, 1]
    return np.expand_dims(arr, axis=0)   # (1, 224, 224, 3)


class DiseaseClassifier:
    """
    Wraps the TFLite interpreter.  Thread-safe (interpreter state is not
    shared across threads; create one instance per worker if needed).

    Singleton pattern is used so the model is loaded once and reused.
    """

    _instance: Optional["DiseaseClassifier"] = None

    @classmethod
    def get(cls) -> "DiseaseClassifier":
        """Return the singleton classifier, loading the model on first call."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._interpreter = None
        self._input_details  = None
        self._output_details = None
        self._load_model()

    def _load_model(self) -> None:
        if not _TFLITE_OK:
            logger.warning("Running without TFLite – predictions will be random stubs.")
            return

        if not MODEL_PATH.exists():
            logger.error("Model file not found at %s.  "
                         "Download it or run scripts/download_model.sh",
                         MODEL_PATH)
            return

        try:
            self._interpreter = tflite.Interpreter(
                model_path=str(MODEL_PATH),
                num_threads=2,   # use 2 of 4 cores to leave headroom
            )
            self._interpreter.allocate_tensors()
            self._input_details  = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
            logger.info("TFLite model loaded: %s  (input=%s)",
                        MODEL_PATH.name,
                        self._input_details[0]["shape"])
        except Exception as exc:
            logger.error("Failed to load TFLite model: %s", exc)
            self._interpreter = None

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, image: np.ndarray) -> Optional[PredictionResult]:
        """
        Run disease classification on a BGR numpy image.

        Returns PredictionResult or None if inference fails / model absent.
        """
        if self._interpreter is None:
            return self._stub_predict()   # fallback for dev/testing

        try:
            tensor = _preprocess(image)
            self._interpreter.set_tensor(
                self._input_details[0]["index"], tensor)
            self._interpreter.invoke()

            output = self._interpreter.get_tensor(
                self._output_details[0]["index"])[0]  # shape: (num_classes,)

            return self._parse_output(output)

        except Exception as exc:
            logger.error("Inference error: %s", exc)
            return None

    def _parse_output(self, scores: np.ndarray) -> PredictionResult:
        labels = AI_CFG.LABELS
        # Top-k indices
        top_indices = np.argsort(scores)[::-1][:AI_CFG.TOP_K]
        top_k       = [(labels[i], float(scores[i])) for i in top_indices]

        best_idx    = top_indices[0]
        best_label  = labels[best_idx]
        best_score  = float(scores[best_idx])
        is_healthy  = best_label == "Healthy"

        severity    = "none" if is_healthy else \
                      _severity_from_confidence(best_score)

        return PredictionResult(
            disease    = best_label,
            confidence = round(best_score, 4),
            severity   = severity,
            top_k      = top_k,
            is_healthy = is_healthy,
        )

    # ── Stub (no model) ───────────────────────────────────────────────────────

    def _stub_predict(self) -> PredictionResult:
        """Return a fake result for development/testing without hardware."""
        import random
        label = random.choice(AI_CFG.LABELS)
        score = round(random.uniform(0.55, 0.99), 4)
        return PredictionResult(
            disease    = label,
            confidence = score,
            severity   = "none" if label == "Healthy" else
                          _severity_from_confidence(score),
            top_k      = [(label, score)],
            is_healthy = label == "Healthy",
        )

    def meets_threshold(self, result: PredictionResult) -> bool:
        """True when confidence is high enough to act on."""
        return result.confidence >= AI_CFG.CONFIDENCE_THRESH
