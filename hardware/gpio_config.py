"""
hardware/gpio_config.py

GPIO initialisation and teardown helpers.

All GPIO interaction goes through this module.  The AI layer NEVER
imports RPi.GPIO directly – it sends JSON actions to the control
layer, which calls these functions.

On non-RPi systems (dev laptops) RPi.GPIO is stubbed out so the rest
of the code can still be imported and unit-tested.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Graceful stub for non-Raspberry-Pi environments ──────────────────────────
try:
    import RPi.GPIO as GPIO          # type: ignore[import]
    _REAL_GPIO = True
except (ImportError, RuntimeError):
    logger.warning("RPi.GPIO not found – running in SIMULATION mode.")
    _REAL_GPIO = False

    class _StubGPIO:
        """Minimal GPIO stub so the code loads on dev machines."""
        BCM = OUT = IN = HIGH = LOW = BOARD = 0
        PUD_UP = PUD_DOWN = 1

        def setmode(self, *a, **kw):          pass
        def setup(self, *a, **kw):            pass
        def output(self, *a, **kw):           pass
        def input(self, pin):                 return 0
        def cleanup(self):                    pass
        def setwarnings(self, flag):          pass

        class PWM:
            def __init__(self, pin, freq):    pass
            def start(self, dc):              pass
            def ChangeDutyCycle(self, dc):    pass
            def stop(self):                   pass

    GPIO = _StubGPIO()  # type: ignore[assignment]


# ── Public helpers ────────────────────────────────────────────────────────────

def init_gpio() -> None:
    """Set BCM mode and configure all pins used by the rover."""
    from config import MOTOR_PINS, SENSOR_PINS

    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)

    # Motor driver pins
    for pin in (
        MOTOR_PINS.ENA, MOTOR_PINS.IN1, MOTOR_PINS.IN2,
        MOTOR_PINS.ENB, MOTOR_PINS.IN3, MOTOR_PINS.IN4,
    ):
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)

    # Ultrasonic sensor
    GPIO.setup(SENSOR_PINS.TRIG, GPIO.OUT)
    GPIO.setup(SENSOR_PINS.ECHO, GPIO.IN)
    GPIO.output(SENSOR_PINS.TRIG, GPIO.LOW)

    # Optional IR sensors (pulled-up internally)
    GPIO.setup(SENSOR_PINS.IR_LEFT,  GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(SENSOR_PINS.IR_RIGHT, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    logger.info("GPIO initialised (real=%s)", _REAL_GPIO)


def cleanup_gpio() -> None:
    """Release all GPIO resources on shutdown."""
    GPIO.cleanup()
    logger.info("GPIO cleaned up.")


def get_gpio() -> "GPIO":          # type: ignore[name-defined]
    """Return the GPIO module (real or stub)."""
    return GPIO


def is_real_hardware() -> bool:
    """True when running on actual Raspberry Pi hardware."""
    return _REAL_GPIO
