#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║   Crop Disease Detector — Raspberry Pi 4 + Kindwise API     ║
║   API  : https://crop.kindwise.com/api/v1/identification    ║
║   Docs : https://crop.kindwise.com/docs                     ║
╚══════════════════════════════════════════════════════════════╝

How to get your FREE API key:
  1. Go to  https://admin.kindwise.com
  2. Sign up (free, no credit card)
  3. Copy your API key and paste below

Install dependencies:
  sudo apt install -y python3-picamera2
  pip install requests pillow

Run:
  python3 crop_disease_kindwise.py                  # capture & analyse once
  python3 crop_disease_kindwise.py --image leaf.jpg # use existing image
  python3 crop_disease_kindwise.py --loop 30        # capture every 30 seconds
  python3 crop_disease_kindwise.py --help
"""

import argparse
import base64
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

# ════════════════════════════════════════════════════
#  ★  PASTE YOUR API KEY HERE  ★
# ════════════════════════════════════════════════════
API_KEY = "lHSHC5L0xnCbB12hHTIjPtxnaKXluDHKto8Wf9nAbb8wdOqInp"

# ════════════════════════════════════════════════════
#  API SETTINGS
# ════════════════════════════════════════════════════
API_URL = "https://crop.kindwise.com/api/v1/identification"

# Which extra details to fetch (comma-separated)
# Options: taxonomy, wiki_url, description, treatment, wiki_description,
#          eppo_code, cause, classification
DETAILS = "taxonomy,description,treatment,wiki_url"

# ════════════════════════════════════════════════════
#  CAMERA SETTINGS
# ════════════════════════════════════════════════════
CAPTURE_WIDTH  = 1920
CAPTURE_HEIGHT = 1080
CAPTURE_PATH   = Path("/tmp/kindwise_capture.jpg")

# ════════════════════════════════════════════════════
#  OUTPUT SETTINGS
# ════════════════════════════════════════════════════
RESULTS_DIR  = Path("./results")     # JSON results and captured images saved here
SAVE_RESULTS = True                  # set False to disable saving

# ════════════════════════════════════════════════════
#  LOGGING
# ════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ────────────────────────────────────────────────────
#  CAMERA  — tries picamera2 first, falls back to
#             legacy picamera, then raises a clear error
# ────────────────────────────────────────────────────

def capture_image(output_path: Path) -> Path:
    """Capture a JPEG from the Pi Camera and save it to output_path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── picamera2 (RPi OS Bullseye / Bookworm, recommended) ──────────────
    try:
        from picamera2 import Picamera2

        log.info("📷 Capturing with picamera2...")
        cam = Picamera2()
        config = cam.create_still_configuration(
            main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT), "format": "RGB888"}
        )
        cam.configure(config)
        cam.start()
        time.sleep(2)                           # let auto-exposure settle
        cam.capture_file(str(output_path))
        cam.stop()
        cam.close()
        log.info("✅ Image captured: %s", output_path)
        return output_path

    except ImportError:
        log.warning("picamera2 not found — trying legacy picamera...")

    # ── legacy picamera (RPi OS Buster / older) ───────────────────────────
    try:
        import picamera

        log.info("📷 Capturing with picamera (legacy)...")
        with picamera.PiCamera() as cam:
            cam.resolution = (CAPTURE_WIDTH, CAPTURE_HEIGHT)
            cam.start_preview()
            time.sleep(2)
            cam.capture(str(output_path))
        log.info("✅ Image captured: %s", output_path)
        return output_path

    except ImportError:
        pass

    # ── neither available ─────────────────────────────────────────────────
    print("\n❌  No camera library found.")
    print("    Install picamera2:  sudo apt install -y python3-picamera2")
    print("    Or pass an image:   python3 crop_disease_kindwise.py --image leaf.jpg\n")
    sys.exit(1)


# ────────────────────────────────────────────────────
#  SAVE CAPTURED IMAGE
# ────────────────────────────────────────────────────

def save_captured_image(src: Path) -> Path:
    """
    Copy the freshly-captured image from /tmp into the results folder
    with a timestamp so it is never overwritten by the next scan.

    Returns the permanent destination path.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = RESULTS_DIR / f"capture_{ts}.jpg"
    shutil.copy2(src, dest)
    log.info("📸 Captured image saved: %s", dest)
    return dest


# ────────────────────────────────────────────────────
#  IMAGE HELPERS
# ────────────────────────────────────────────────────

def image_to_base64(path: Path) -> str:
    """Read image file → base64 string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def resize_image(path: Path, max_long_side: int = 800) -> Path:
    """
    Kindwise recommends the longer side be ~800 px for best speed.
    Resize in-place if needed (requires Pillow).
    """
    try:
        from PIL import Image

        img = Image.open(path)
        w, h = img.size
        long_side = max(w, h)

        if long_side <= max_long_side:
            return path                         # already small enough

        scale  = max_long_side / long_side
        new_w  = int(w * scale)
        new_h  = int(h * scale)
        img    = img.resize((new_w, new_h), Image.LANCZOS)
        img.save(path, "JPEG", quality=90)
        log.info("🔧 Resized image to %d×%d px", new_w, new_h)

    except ImportError:
        log.warning("Pillow not installed — sending full-size image (may be slower).")

    return path


# ────────────────────────────────────────────────────
#  KINDWISE API CALL
# ────────────────────────────────────────────────────

def analyse(image_path: Path) -> dict:
    """
    POST the image to Kindwise crop.health API and return the JSON response.

    Request
    ───────
    POST https://crop.kindwise.com/api/v1/identification
    Headers : Api-Key: <key>
              Content-Type: application/json
    Body    : { "images": ["<base64>"], "similar_images": true }
    Params  : details=taxonomy,description,treatment,wiki_url
              language=en

    Response (simplified)
    ─────────────────────
    {
      "result": {
        "crop": {
          "suggestions": [
            { "name": "Tomato", "probability": 0.98, ... }
          ]
        },
        "disease": {
          "suggestions": [
            {
              "name": "Late Blight",
              "probability": 0.91,
              "details": {
                "description": "...",
                "treatment": {
                  "biological": [...],
                  "chemical": [...],
                  "prevention": [...]
                },
                "wiki_url": "https://..."
              }
            }
          ],
          "is_healthy": false
        }
      }
    }
    """
    if API_KEY in ("", "YOUR_API_KEY_HERE"):
        print("\n❌  API key not set!")
        print("    Edit API_KEY in this script, OR")
        print("    Run:  export KINDWISE_API_KEY=your_key_here\n")
        sys.exit(1)

    image_path = resize_image(image_path)
    b64_image  = image_to_base64(image_path)

    headers = {
        "Api-Key":      API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "images":         [b64_image],
        "similar_images": True,             # get reference comparison images
    }
    params = {
        "details":  DETAILS,
        "language": "en",
    }

    log.info("🌐 Sending image to Kindwise crop.health API...")
    try:
        resp = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            params=params,
            timeout=60,
        )
        resp.raise_for_status()

    except requests.exceptions.HTTPError as e:
        log.error("HTTP %s — %s", e.response.status_code, e.response.text)
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        log.error("Cannot connect to %s — check your internet connection.", API_URL)
        sys.exit(1)
    except requests.exceptions.Timeout:
        log.error("Request timed out (60 s). Try again.")
        sys.exit(1)

    return resp.json()


# ────────────────────────────────────────────────────
#  DISPLAY RESULTS
# ────────────────────────────────────────────────────

def display(data: dict) -> None:
    """Print a clean, human-readable summary of the API response."""
    result  = data.get("result", {})
    crop    = result.get("crop",    {})
    disease = result.get("disease", {})

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║        KINDWISE CROP DISEASE ANALYSIS            ║")
    print("╚══════════════════════════════════════════════════╝")

    # ── Crop identification ───────────────────────────────────────────────
    crop_suggestions = crop.get("suggestions", [])
    if crop_suggestions:
        top_crop = crop_suggestions[0]
        print(f"\n🌱  Crop detected : {top_crop['name']}  "
              f"({top_crop['probability'] * 100:.1f}% confidence)")

        if len(crop_suggestions) > 1:
            others = ", ".join(
                f"{s['name']} ({s['probability']*100:.0f}%)"
                for s in crop_suggestions[1:3]
            )
            print(f"    Other options  : {others}")
    else:
        print("\n🌱  Crop : Could not identify the crop.")

    # ── Health status ─────────────────────────────────────────────────────
    is_healthy = disease.get("is_healthy", None)
    if is_healthy is True:
        print("\n✅  Plant health   : HEALTHY — no disease detected.")
        print("═" * 52)
        return
    elif is_healthy is False:
        print("\n⚠️   Plant health   : DISEASE DETECTED")
    else:
        print("\n❓  Plant health   : Unknown")

    # ── Disease details ───────────────────────────────────────────────────
    disease_suggestions = disease.get("suggestions", [])
    if not disease_suggestions:
        print("    No specific disease suggestions returned.")
        print("═" * 52)
        return

    print(f"\n{'─'*52}")
    print("  TOP DISEASE DIAGNOSES")
    print(f"{'─'*52}")

    for i, s in enumerate(disease_suggestions[:3], start=1):
        name    = s.get("name", "Unknown")
        prob    = s.get("probability", 0) * 100
        details = s.get("details", {})

        print(f"\n  [{i}] {name}")
        print(f"      Confidence  : {prob:.1f}%")

        # Description
        desc = details.get("description") or details.get("wiki_description")
        if desc:
            # Trim to 3 sentences for readability
            sentences = desc.replace("\n", " ").split(". ")
            short_desc = ". ".join(sentences[:3]).strip()
            if not short_desc.endswith("."):
                short_desc += "."
            print(f"      Description : {short_desc}")

        # Taxonomy
        taxonomy = details.get("taxonomy", {})
        if taxonomy:
            genus   = taxonomy.get("genus",   "")
            species = taxonomy.get("species", "")
            if genus or species:
                print(f"      Scientific  : {genus} {species}".strip())

        # Treatment
        treatment = details.get("treatment", {})
        if treatment:
            print("      Treatment   :")
            for category in ("prevention", "biological", "chemical"):
                tips = treatment.get(category, [])
                if tips:
                    print(f"        {category.capitalize()}:")
                    for tip in tips[:2]:          # show top 2 tips per category
                        print(f"          • {tip}")

        # Wikipedia link
        wiki = details.get("wiki_url")
        if wiki:
            print(f"      More info   : {wiki}")

        # Similar reference images
        similar = s.get("similar_images", [])
        if similar:
            print(f"      Ref. images : {similar[0].get('url', '')}")

        print(f"      {'─'*44}")

    print("═" * 52)
    print()


# ────────────────────────────────────────────────────
#  SAVE RESULTS TO DISK
# ────────────────────────────────────────────────────

def save_result(data: dict, image_path: Path) -> None:
    """Save the full JSON response to the results folder."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = RESULTS_DIR / f"result_{ts}.json"
    payload  = {
        "timestamp":  ts,
        "image_path": str(image_path),
        "response":   data,
    }
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("💾 Result saved: %s", out_file)


# ────────────────────────────────────────────────────
#  MAIN FLOW
# ────────────────────────────────────────────────────

def run_once(image_arg: str | None) -> None:
    """Capture (or load) one image → save → analyse → display → save result."""
    if image_arg:
        image_path = Path(image_arg)
        if not image_path.exists():
            print(f"\n❌  File not found: {image_path}\n")
            sys.exit(1)
        log.info("📂 Using existing image: %s", image_path)
    else:
        image_path = capture_image(CAPTURE_PATH)
        # ── Save captured image before analysis so it is never lost ──────
        if SAVE_RESULTS:
            image_path = save_captured_image(image_path)

    data = analyse(image_path)
    display(data)

    if SAVE_RESULTS:
        save_result(data, image_path)


def run_loop(interval: int, image_arg: str | None) -> None:
    """Continuously capture and analyse at a fixed interval."""
    log.info("🔄 Loop mode — every %d seconds. Press Ctrl+C to stop.", interval)
    try:
        count = 1
        while True:
            print(f"\n{'━'*52}")
            print(f"  Scan #{count}  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'━'*52}")
            run_once(image_arg)
            count += 1
            log.info("⏳ Waiting %d s for next scan...", interval)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n👋 Stopped by user.")


# ────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────

def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Raspberry Pi 4 crop disease detector using Kindwise crop.health API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 crop_disease_kindwise.py
  python3 crop_disease_kindwise.py --image leaf.jpg
  python3 crop_disease_kindwise.py --loop 30
  python3 crop_disease_kindwise.py --key MY_API_KEY --image test.jpg
        """,
    )
    p.add_argument(
        "--image", "-i",
        metavar="PATH",
        help="Path to an existing image file (skip camera capture)",
    )
    p.add_argument(
        "--loop", "-l",
        metavar="SECONDS",
        type=int,
        default=0,
        help="Keep scanning at this interval in seconds (0 = run once)",
    )
    p.add_argument(
        "--key", "-k",
        metavar="API_KEY",
        help="Kindwise API key (overrides env var and script default)",
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save results or captured images to disk",
    )
    return p.parse_args()


def main() -> None:
    args = build_args()

    # Allow key override from CLI
    if args.key:
        global API_KEY
        API_KEY = args.key

    # Allow disabling save
    global SAVE_RESULTS
    if args.no_save:
        SAVE_RESULTS = False

    if args.loop > 0:
        run_loop(args.loop, args.image)
    else:
        run_once(args.image)


if __name__ == "__main__":
    main()