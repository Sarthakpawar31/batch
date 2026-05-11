"""
hardware/dht11.py

DHT11 temperature + humidity sensor driver.

This implementation uses the Pi's GPIO pin directly so there is no
additional sensor library dependency.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from config import SENSOR_PINS
from hardware.gpio_config import get_gpio

logger = logging.getLogger(__name__)
GPIO = get_gpio()

# DHT11 timing thresholds
_START_SIGNAL_S = 0.020
_RESPONSE_TIMEOUT_S = 0.005
_BIT_TIMEOUT_S = 0.005
_HIGH_PULSE_THRESHOLD_NS = 50_000  # 50 µs


def _bits_to_byte(bits: list[int]) -> int:
    return sum(1 << (7 - idx) for idx, bit in enumerate(bits) if bit)


def _wait_for(pin: int, state: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if GPIO.input(pin) == state:
            return True
    return False


def _measure_high_pulse(pin: int) -> Optional[int]:
    if not _wait_for(pin, GPIO.HIGH, _BIT_TIMEOUT_S):
        return None
    start_ns = time.monotonic_ns()
    if not _wait_for(pin, GPIO.LOW, _BIT_TIMEOUT_S):
        return None
    return time.monotonic_ns() - start_ns


class DHT11Reader:
    """Read temperature and humidity from a DHT11 sensor."""

    def __init__(self) -> None:
        self._pin = SENSOR_PINS.DHT11

    def read(self, retries: int = 3) -> tuple[Optional[float], Optional[float]]:
        """Read sensor values. Returns (temperature_c, humidity_pct)."""
        for attempt in range(1, retries + 1):
            temperature, humidity = self._read_once()
            if temperature is not None and humidity is not None:
                return temperature, humidity
            logger.debug("DHT11 read attempt %d failed", attempt)
            time.sleep(0.5)
        logger.warning("DHT11 sensor read failed after %d attempts", retries)
        return None, None

    def _read_once(self) -> tuple[Optional[float], Optional[float]]:
        try:
            GPIO.setup(self._pin, GPIO.OUT)
            GPIO.output(self._pin, GPIO.LOW)
            time.sleep(_START_SIGNAL_S)
            GPIO.output(self._pin, GPIO.HIGH)
            time.sleep(0.00004)
            GPIO.setup(self._pin, GPIO.IN, pull_up_down=GPIO.PUD_OFF)

            if not _wait_for(self._pin, GPIO.LOW, _RESPONSE_TIMEOUT_S):
                return None, None
            if not _wait_for(self._pin, GPIO.HIGH, _RESPONSE_TIMEOUT_S):
                return None, None
            if not _wait_for(self._pin, GPIO.LOW, _RESPONSE_TIMEOUT_S):
                return None, None

            bits: list[int] = []
            for _ in range(40):
                duration_ns = _measure_high_pulse(self._pin)
                if duration_ns is None:
                    return None, None
                bits.append(1 if duration_ns > _HIGH_PULSE_THRESHOLD_NS else 0)

            raw = [
                _bits_to_byte(bits[i : i + 8])
                for i in range(0, 40, 8)
            ]
            checksum = sum(raw[:4]) & 0xFF
            if checksum != raw[4]:
                logger.debug(
                    "DHT11 checksum mismatch: expected=%d got=%d",
                    checksum,
                    raw[4],
                )
                return None, None

            humidity = float(raw[0])
            temperature = float(raw[2])
            logger.debug(
                "DHT11 read success: temperature=%.1f°C humidity=%.1f%%",
                temperature,
                humidity,
            )
            return temperature, humidity

        except Exception as exc:  # pragma: no cover
            logger.exception("DHT11 read error: %s", exc)
            return None, None
