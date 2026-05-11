"""
hardware/ultrasonic.py

HC-SR04 ultrasonic distance sensor driver.

Timing diagram:
  TRIG  ──[10µs pulse]──►
  ECHO  ◄──[pulse width proportional to distance]──

distance_cm = (echo_duration_s × 34300) / 2
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from config import SENSOR_PINS, MOTOR_CFG
from hardware.gpio_config import get_gpio

logger = logging.getLogger(__name__)
GPIO = get_gpio()

# Speed of sound at ~20 °C in cm/s
_SOUND_CM_S = 34300.0
_TRIGGER_S  = 0.00001   # 10 µs trigger pulse
_TIMEOUT_S  = 0.015      # 30 ms → max ~5 m range; avoids blocking


def read_distance_cm() -> Optional[float]:
    """
    Blocking single measurement.

    Returns distance in cm, or None on timeout / error.
    Typical call time < 5 ms.
    """
    try:
        # Trigger pulse
        GPIO.output(SENSOR_PINS.TRIG, GPIO.HIGH)
        time.sleep(_TRIGGER_S)
        GPIO.output(SENSOR_PINS.TRIG, GPIO.LOW)

        # Wait for echo to go HIGH
        t0 = time.monotonic()
        while GPIO.input(SENSOR_PINS.ECHO) == 0:
            if time.monotonic() - t0 > _TIMEOUT_S:
                return None

        pulse_start = time.monotonic()

        # Wait for echo to go LOW
        while GPIO.input(SENSOR_PINS.ECHO) == 1:
            if time.monotonic() - pulse_start > _TIMEOUT_S:
                return None

        pulse_end = time.monotonic()
        duration  = pulse_end - pulse_start
        distance  = (duration * _SOUND_CM_S) / 2.0
        return round(distance, 1)

    except Exception as exc:                       # pragma: no cover
        logger.error("Ultrasonic read error: %s", exc)
        return None


async def async_read_distance_cm() -> Optional[float]:
    """
    Non-blocking wrapper – runs the blocking read in the default executor
    so the asyncio event loop is not stalled.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, read_distance_cm)


def read_ir_sensors() -> tuple[bool, bool]:
    """Read left/right IR proximity sensor states.

    Returns (left_triggered, right_triggered). IR sensors are pulled up,
    so GPIO.LOW means an obstacle is detected.
    """
    try:
        left  = GPIO.input(SENSOR_PINS.IR_LEFT) == GPIO.LOW
        right = GPIO.input(SENSOR_PINS.IR_RIGHT) == GPIO.LOW
        return left, right
    except Exception as exc:                       # pragma: no cover
        logger.error("IR sensor read error: %s", exc)
        return False, False


def is_obstacle_ahead() -> bool:
    """
    Convenience check: True when an obstacle is within the configured
    OBSTACLE_DIST_CM threshold.
    """
    dist = read_distance_cm()
    if dist is None:
        return False   # sensor error → assume clear
    return dist < MOTOR_CFG.OBSTACLE_DIST_CM 