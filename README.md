# AgriRover – Autonomous Agricultural Disease Detection Robot

A production-grade autonomous rover for plant disease detection.
Runs entirely on **Raspberry Pi 4 (1 GB RAM)** using edge AI + optional cloud reasoning.

---

## Architecture Overview

```
SENSE ──────────► THINK ──────────► ACT
Camera + Sensors   TFLite + FSM     Motors + DB + API
```

The rover operates a continuous **OBSERVE → ANALYZE → DECIDE → ACT** loop
with no human interaction required.

---

## Folder Structure

```
robot/
├── ai/
│   ├── classify.py          TFLite inference engine
│   ├── disease_model.tflite MobileNetV2 (you must add this)
│   └── openrouter_client.py Cloud reasoning via OpenRouter
│
├── vision/
│   ├── camera.py            PiCamera2 / OpenCV unified interface
│   └── leaf_detection.py    HSV-based plant detection (no DNN)
│
├── hardware/
│   ├── gpio_config.py       GPIO init/teardown + non-RPi stub
│   ├── motors.py            L298N dual H-bridge driver
│   └── ultrasonic.py        HC-SR04 distance sensor
│
├── memory/
│   └── database.py          Async SQLite via aiosqlite
│
├── control/
│   ├── state_machine.py     8-state FSM (IDLE→EXPLORING→…)
│   └── planner.py           Autonomous decision loop
│
├── scripts/
│   ├── download_model.sh    Downloads placeholder TFLite model
│   └── train_model.py       Fine-tune MobileNetV2 on PlantVillage
│
├── reports/                 Auto-created; stores captured images
│
├── config.py                ALL tuneable parameters
├── main.py                  Entry point
├── requirements.txt
└── .env.example
```

---

## GPIO Wiring

### L298N Motor Driver

```
Raspberry Pi 4          L298N Module
─────────────────────────────────────────────────────────────
GPIO 12  (PWM)  ────►   ENA   (Left  motor speed)
GPIO 20         ────►   IN1   (Left  motor direction A)
GPIO 21         ────►   IN2   (Left  motor direction B)
GPIO 13  (PWM)  ────►   ENB   (Right motor speed)
GPIO 19         ────►   IN3   (Right motor direction A)
GPIO 26         ────►   IN4   (Right motor direction B)
5V                  ────►   +5V  (logic power)
GND                 ────►   GND

L298N OUT1/OUT2 ────►   Left  DC motor terminals
L298N OUT3/OUT4 ────►   Right DC motor terminals
L298N 12V       ────►   Battery+ (7–12 V)
```

### HC-SR04 Ultrasonic Sensor

```
Raspberry Pi 4          HC-SR04
─────────────────────────────────────────────────────────────
GPIO 23         ────►   TRIG
GPIO 24         ────►   ECHO  (via voltage divider: 1kΩ+2kΩ)
3.3V/5V         ────►   VCC
GND             ────►   GND

⚠️  ECHO outputs 5V – use a voltage divider to protect the Pi's
    3.3V GPIO.  1kΩ from ECHO → Pi pin, 2kΩ from Pi pin → GND.
```

### Optional IR Sensors

```
GPIO 17  ────►   IR_LEFT  (signal)
GPIO 27  ────►   IR_RIGHT (signal)
3.3V     ────►   VCC
GND      ────►   GND
```

---

## State Machine

```
          ┌─────────────────────────────────────────────────┐
          │                                                 │
  [IDLE]──►[EXPLORING]──►[OBSTACLE]──────────────────────► │
               │   ▲          └──────────────────────────► │
               │   └──────────────────────────────────────► │
               │                                             │
               └──►[PLANT_FOUND]──►[INSPECTING]──►[REPORTING]
                                        │
                                        └──►[EXPLORING] (resume)

  [EXPLORING]──►[RETURNING]──►[IDLE]  (battery low)
  Any state  ──►[ERROR]                (hardware fault)
```

---

## Detection Pipeline

```
Camera Frame
    │
    ▼
Leaf Detection (HSV + contour analysis, ~5 ms)
    │
    ▼ best leaf ROI (BGR crop)
    │
    ▼
Preprocess (resize 224×224, normalise [-1,1])
    │
    ▼
TFLite Inference (MobileNetV2, ~120 ms on RPi 4)
    │
    ▼
{disease, confidence, severity, top_k}
    │
    ├──► confidence < threshold → log & skip
    │
    └──► confidence ≥ threshold:
            ├── save image to reports/
            ├── persist to SQLite
            └── async OpenRouter call → treatment advice
```

---

## Database Schema

```sql
-- Persistent detection records
CREATE TABLE detections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,     -- ISO-8601 UTC
    disease         TEXT    NOT NULL,
    confidence      REAL    NOT NULL,     -- 0.0–1.0
    severity        TEXT    NOT NULL,     -- none|low|moderate|high
    image_path      TEXT,                 -- path in reports/
    treatment_json  TEXT,                 -- OpenRouter JSON blob
    location_tag    TEXT,                 -- optional GPS / grid
    created_at      TEXT    DEFAULT (datetime('now'))
);

-- Inspection session metadata
CREATE TABLE sessions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    start_time       TEXT NOT NULL,
    end_time         TEXT,
    total_detections INTEGER DEFAULT 0,
    summary          TEXT                 -- AI-generated paragraph
);
```

---

## Setup Instructions

### 1. System packages

```bash
sudo apt update && sudo apt install -y \
    python3-pip python3-venv \
    libopencv-dev python3-opencv \
    libatlas-base-dev            # for numpy on RPi
```

### 2. Virtual environment

```bash
cd ~/robot
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. TFLite runtime (RPi-specific wheel)

```bash
# For RPi 4 (aarch64 / armv7l), tflite-runtime is lighter than full TF:
pip install tflite-runtime
```

### 4. Download / prepare the disease model

```bash
bash scripts/download_model.sh
# This grabs a placeholder MobileNetV2.
# For real disease detection, fine-tune using scripts/train_model.py
# on a desktop GPU, then copy the .tflite to ai/disease_model.tflite.
```

### 5. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

### 6. Run

```bash
python main.py
```

### 7. Autostart on boot (optional)

```bash
# /etc/systemd/system/agri-rover.service
[Unit]
Description=AgriRover
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/robot
ExecStart=/home/pi/robot/.venv/bin/python main.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable agri-rover
sudo systemctl start  agri-rover
```

---

## Performance Optimisation

| Technique | Impact |
|---|---|
| `tflite-runtime` instead of full TF | saves ~800 MB RAM |
| 320×240 camera resolution | 4× fewer pixels vs 640×480 |
| Frame skipping (`FRAME_SKIP=3`) | reduces CPU 60% |
| HSV leaf detection (no DNN) | ~5 ms vs ~200 ms for YOLO |
| `uvloop` event loop | ~2× asyncio throughput |
| `opencv-headless` | no GUI deps, saves ~150 MB |
| `num_threads=2` in TFLite | keeps 2 cores free for I/O |
| aiosqlite async writes | non-blocking DB operations |
| Dynamic-range quantised model | 14 MB vs 55 MB float model |

Target memory footprint: **≤ 250 MB RSS** (plenty of headroom on 1 GB).

---

## OpenRouter Integration

The rover uses OpenRouter **only** for:
- Human-readable disease explanations
- Treatment and prevention advice
- Session summary generation

It does **not** use OpenRouter for movement decisions or real-time control.
The rover functions fully offline if no API key is set.

Recommended model: `mistralai/mistral-7b-instruct`  
Cost: ~$0.0002 per detection event.

---

## Future Scalability

1. **GPS Integration** – add a U-Blox NEO-6M for grid-based field mapping
2. **ROS 2** – wrap each module as a ROS 2 node once field testing is complete
3. **SLAM** – add a lidar (RPLidar A1) for map-based navigation
4. **Fleet** – multiple rovers reporting to a central MQTT broker
5. **Better model** – EfficientNet-Lite4 for higher accuracy (~+5%) with only +20 MB
6. **Camera upgrade** – Pi HQ camera for better leaf macro shots
7. **Solar charging** – MPPT charger + LiFePO4 cells for all-day operation

---

## AI Architecture – No Chatbot, Pure Autonomy

```
Sensor layer  →  Event bus (asyncio queues)
                     │
              Decision Planner (FSM-driven)
                     │
          ┌──────────┴──────────┐
    Edge AI (TFLite)     Cloud AI (OpenRouter)
    ────────────────     ─────────────────────
    •  Leaf detection     •  Treatment text
    •  Disease class      •  Session report
    •  Confidence score   •  (optional only)
          │
    Motor commands (JSON → GPIO abstraction layer)
```

The AI layer **never** touches GPIO.  
Movement is controlled exclusively by the `MotorController` class,  
called only from `planner.py`.
