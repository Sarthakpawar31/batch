"""
config.py – Central configuration for the Agricultural Rover.

All tuneable parameters live here.  Hardware pin numbers, model paths,
API keys, and timing constants are kept separate from business logic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads .env in project root

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "ai" / "disease_model.tflite"
DB_PATH    = BASE_DIR / "memory" / "rover.db"
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)


# ── GPIO Pin Assignments (BCM numbering) ──────────────────────────────────────
@dataclass(frozen=True)
class MotorPins:
    # Left motor  (Motor A on L298N)
    ENA: int = 12   # PWM-capable pin
    IN1: int = 20
    IN2: int = 21
    # Right motor (Motor B on L298N)
    ENB: int = 13   # PWM-capable pin
    IN3: int = 19
    IN4: int = 26

@dataclass(frozen=True)
class SensorPins:
    TRIG:        int = 23   # Ultrasonic trigger
    ECHO:        int = 24   # Ultrasonic echo
    IR_LEFT:     int = 17   # Optional IR sensor (left)
    IR_RIGHT:    int = 27   # Optional IR sensor (right)
    DHT11:       int = 4    # Temperature / humidity sensor

@dataclass(frozen=True)
class EnvSensorPins:
    GPS_RX:      int = 15   # UART RX (GPS TX → Pi RX)
    GPS_TX:      int = 14   # UART TX (GPS RX → Pi TX)
    I2C_SDA:     int = 2    # Compass / I2C SDA
    I2C_SCL:     int = 3    # Compass / I2C SCL

@dataclass(frozen=True)
class GPSConfig:
    DEVICE: str = os.getenv("GPS_DEVICE", "/dev/serial0")
    BAUDRATE: int = int(os.getenv("GPS_BAUDRATE", "9600"))
    TIMEOUT: float = 1.0

@dataclass(frozen=True)
class CompassConfig:
    I2C_BUS: int = int(os.getenv("I2C_BUS", "1"))
    I2C_ADDRESS: int = int(os.getenv("COMPASS_ADDRESS", "0x1E"), 0)

MOTOR_PINS  = MotorPins()
SENSOR_PINS = SensorPins()
ENV_PINS    = EnvSensorPins()
GPS_CFG     = GPSConfig()
COMPASS_CFG = CompassConfig()


# ── Motor / Movement ──────────────────────────────────────────────────────────
@dataclass(frozen=True)
class MotorConfig:
    PWM_FREQ:          int   = 1000   # Hz
    DEFAULT_SPEED:     int   = 35     # % duty cycle  (0–100)
    TURN_SPEED:        int   = 10
    OBSTACLE_DIST_CM:  float = 35.0   # stop if closer than this
    CLEAR_DIST_CM:     float = 45.0   # require this to be clear before resuming
    REVERSE_SECS:      float = 0.6
    TURN_SECS:         float = 1.0

MOTOR_CFG = MotorConfig()


# ── Camera / Vision ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CameraConfig:
    WIDTH:          int   = 320      # inference resolution – keep low!
    HEIGHT:         int   = 240
    FRAMERATE:      int   = 10
    FRAME_SKIP:     int   = 1        # process every Nth frame
    LEAF_MIN_AREA:  int   = 2000     # px² – ignore tiny blobs
    USE_PICAMERA:   bool  = True     # False → USB webcam via OpenCV

CAM_CFG = CameraConfig()


# ── AI / Inference ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class AIConfig:
    INPUT_SIZE:         int   = 224   # MobileNetV2 expects 224×224
    CONFIDENCE_THRESH:  float = 0.60  # min confidence to report a disease
    TOP_K:              int   = 3     # how many top classes to keep

    # Disease class labels (must match model's output order)
    LABELS: tuple = (
        "Healthy",
        "Early Blight",
        "Late Blight",
        "Leaf Mold",
        "Septoria Leaf Spot",
        "Spider Mites",
        "Target Spot",
        "Yellow Leaf Curl Virus",
        "Mosaic Virus",
        "Bacterial Spot",
    )

AI_CFG = AIConfig()


# ── OpenRouter API ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class APIConfig:
    BASE_URL:   str = "https://openrouter.ai/api/v1/chat/completions"
    API_KEY:    str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    MODEL:      str = "mistralai/mistral-7b-instruct"   # cheap + fast
    TIMEOUT:    int = 20    # seconds
    MAX_TOKENS: int = 400

API_CFG = APIConfig()


# ── Autonomy Loop Timing ──────────────────────────────────────────────────────
@dataclass(frozen=True)
class LoopConfig:
    SENSOR_HZ:       float = 10.0   # ultrasonic poll rate
    ENV_HZ:          float = 0.2    # slower environment sensor poll rate
    VISION_HZ:       float = 5.0    # leaf-detection rate
    DECISION_HZ:     float = 2.0    # planner tick rate
    BATTERY_CHECK_S: int   = 60     # seconds between battery checks
    BATTERY_LOW_V:   float = 6.5    # volts (read via ADC if available)

LOOP_CFG = LoopConfig()


# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL  = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s – %(message)s"
