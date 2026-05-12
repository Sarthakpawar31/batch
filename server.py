from __future__ import annotations

import base64
import io
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from datetime import datetime

from ai.classify import DiseaseClassifier, PredictionResult
from config import AI_CFG, REPORT_DIR

logger = logging.getLogger(__name__)


DISEASE_CURES: Dict[str, str] = {
    "Healthy": "No treatment needed. Continue good cultural practices.",
    "Early Blight": "Remove infected leaves, improve airflow, apply appropriate fungicide.",
    "Late Blight": "Remove affected plants, apply copper-based fungicides, avoid overhead watering.",
    "Leaf Mold": "Improve ventilation, reduce humidity, remove infected foliage.",
    "Septoria Leaf Spot": "Remove debris, apply protective fungicides, rotate crops.",
    "Spider Mites": "Wash leaves, introduce predators (e.g. predatory mites), use miticide if severe.",
    "Target Spot": "Remove infected tissue, use fungicide sprays and crop rotation.",
    "Yellow Leaf Curl Virus": "No cure—remove infected plants and control whitefly vectors.",
    "Mosaic Virus": "No chemical cure—remove infected plants and practice sanitation.",
    "Bacterial Spot": "Remove infected leaves, copper sprays can reduce spread; use clean seed.",
}


# Optional external API analyser (Kindwise / Plantix wrapper)
_USE_KINDWISE = False
_KINDWISE = None
try:
    import ai.plantix_disease_analyzer as kindwise  # type: ignore
    # Allow env override for API key used by that module
    env_key = os.getenv("KINDWISE_API_KEY")
    if env_key:
        try:
            kindwise.API_KEY = env_key  # type: ignore
        except Exception:
            pass

    api_key = getattr(kindwise, "API_KEY", "")
    if api_key and api_key not in ("", "YOUR_API_KEY_HERE"):
        _USE_KINDWISE = True
        _KINDWISE = kindwise
        logger.info("Using Kindwise analyser (ai/plantix_disease_analyzer.py)")
    else:
        logger.info("Kindwise analyser present but API key not set; skipping")
except Exception:
    logger.info("Kindwise analyser not available; using local model only")


def make_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB upload limit
    app.secret_key = "replace-with-secure-key"


    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html")


    @app.route("/predict", methods=["POST"])
    def predict():
        file = request.files.get("image")
        if file is None or file.filename == "":
            return redirect(url_for("index"))

        data = file.read()
        # decode image bytes to BGR numpy array
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return render_template("result.html", error="Uploaded file is not a valid image.")

        # Save uploaded image to a temporary file for external API if needed
        tmp_path = None
        api_used = "local"
        if _USE_KINDWISE and _KINDWISE is not None:
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            try:
                tf.write(data)
                tf.flush()
                tmp_path = Path(tf.name)
            finally:
                tf.close()

            try:
                resp = _KINDWISE.analyse(tmp_path)  # returns dict
                # Parse Kindwise response
                disease = None
                confidence = None
                cure_text = None
                result_block = resp.get("result", {}).get("disease", {})
                if result_block.get("is_healthy") is True:
                    disease = "Healthy"
                    confidence = 1.0
                else:
                    suggestions = result_block.get("suggestions", [])
                    if suggestions:
                        top = suggestions[0]
                        disease = top.get("name")
                        confidence = float(top.get("probability", 0.0))
                        details = top.get("details", {}) or {}
                        treatment = details.get("treatment", {}) or {}
                        # Summarize treatment categories into a short text
                        parts = []
                        for k in ("prevention", "biological", "chemical"):
                            tips = treatment.get(k, [])
                            if tips:
                                parts.append(f"{k.capitalize()}: " + "; ".join(tips[:2]))
                        if parts:
                            cure_text = " ".join(parts)

                if disease is not None:
                    api_used = "kindwise"
                    # prepare top_k as best-effort
                    top_k = []
                    suggestions = result_block.get("suggestions", [])
                    for s in suggestions[:3]:
                        top_k.append((s.get("name", ""), float(s.get("probability", 0.0))))

                    context = {
                        "disease": disease,
                        "confidence": round(confidence, 4) if confidence is not None else 0.0,
                        "severity": ("none" if disease == "Healthy" else "moderate"),
                        "top_k": top_k,
                        "is_healthy": disease == "Healthy",
                        "cure": cure_text or DISEASE_CURES.get(disease, "No treatment information available."),
                        "image_data": f"data:image/jpeg;base64,{base64.b64encode(data).decode('ascii')}",
                    }
                    context["source"] = "Kindwise API"
                    return render_template("result.html", **context)


                    @app.route("/reports/<path:filename>")
                    def reports_file(filename: str):
                        """Serve image files from the REPORT_DIR folder."""
                        return send_from_directory(str(REPORT_DIR), filename)


                    @app.route("/gallery")
                    def gallery():
                        """Render a simple gallery of images found in REPORT_DIR.

                        Expects image filenames like: 20260510_172100_Bacterial_Spot.jpg
                        Parses the timestamp and disease from the filename and looks up the cure.
                        """
                        items = []
                        if REPORT_DIR.exists():
                            for entry in sorted(REPORT_DIR.iterdir(), reverse=True):
                                if not entry.is_file():
                                    continue
                                if entry.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                                    continue

                                name = entry.name
                                # parse date and disease from filename
                                stem = entry.stem
                                parts = stem.split("_", 2)
                                date_str = None
                                disease = "Unknown"
                                date_obj = None
                                if len(parts) >= 2:
                                    date_str = f"{parts[0]}_{parts[1]}"
                                    try:
                                        date_obj = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
                                    except Exception:
                                        date_obj = None
                                if len(parts) == 3:
                                    disease = parts[2].replace("_", " ")

                                items.append({
                                    "filename": name,
                                    "url": url_for("reports_file", filename=name),
                                    "disease": disease,
                                    "date": date_obj.strftime("%Y-%m-%d %H:%M:%S") if date_obj else "Unknown",
                                    "cure": DISEASE_CURES.get(disease, "No treatment information available."),
                                })

                        return render_template("gallery.html", items=items)

            except Exception as exc:
                logger.exception("Kindwise analyse failed: %s", exc)
                # fall through to local model
            finally:
                # clean up temp file
                try:
                    if tmp_path and tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass

        # Fallback: local classifier
        classifier = DiseaseClassifier.get()
        result: PredictionResult | None = classifier.predict(img)

        if result is None:
            return render_template("result.html", error="Model unavailable or inference failed.")

        cure = DISEASE_CURES.get(result.disease, "No treatment information available.")

        # prepare image data URL for display
        b64 = base64.b64encode(data).decode("ascii")
        data_url = f"data:image/jpeg;base64,{b64}"

        context = {
            "disease": result.disease,
            "confidence": result.confidence,
            "severity": result.severity,
            "top_k": result.top_k,
            "is_healthy": result.is_healthy,
            "cure": cure,
            "image_data": data_url,
            "source": "Local model",
        }
        return render_template("result.html", **context)


    return app


if __name__ == "__main__":
    app = make_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
