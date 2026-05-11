"""
hardware/gps.py

GPS NMEA parsing support for a UART-based GPS module.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from config import GPS_CFG

logger = logging.getLogger(__name__)

try:
    import serial  # type: ignore[import]
except ImportError:  # pragma: no cover
    serial = None


def _parse_latlon(raw_value: str, hemisphere: str) -> Optional[float]:
    if not raw_value or not hemisphere:
        return None

    try:
        if hemisphere in ("N", "S"):
            degrees = int(raw_value[:2])
            minutes = float(raw_value[2:])
        else:
            degrees = int(raw_value[:3])
            minutes = float(raw_value[3:])
        value = degrees + minutes / 60.0
        if hemisphere in ("S", "W"):
            value = -value
        return value
    except ValueError:
        return None


class GPSReader:
    """Simple GPS reader that parses NMEA sentences from a serial device."""

    def __init__(self) -> None:
        self.device = GPS_CFG.DEVICE
        self.baudrate = GPS_CFG.BAUDRATE
        self.timeout = GPS_CFG.TIMEOUT
        self._port: Optional[Any] = None

    def _ensure_port(self) -> bool:
        if serial is None:
            logger.warning("pyserial not installed; GPS support disabled.")
            return False
        if self._port is not None:
            return True
        try:
            self._port = serial.Serial(
                port=self.device,
                baudrate=self.baudrate,
                timeout=self.timeout,
            )
            logger.debug("GPS serial port opened: %s", self.device)
            return True
        except Exception as exc:  # pragma: no cover
            logger.warning("Unable to open GPS port %s: %s", self.device, exc)
            self._port = None
            return False

    def read_location(self) -> Optional[Dict[str, Any]]:
        if not self._ensure_port():
            return None

        try:
            deadline = time.time() + self.timeout
            while time.time() < deadline:
                raw = self._port.readline()
                if not raw:
                    continue
                try:
                    sentence = raw.decode("ascii", errors="ignore").strip()
                except Exception:
                    continue
                if sentence.startswith("$GPRMC") or sentence.startswith("$GNRMC"):
                    location = self._parse_rmc(sentence)
                    if location:
                        return location
                elif sentence.startswith("$GPGGA") or sentence.startswith("$GNGGA"):
                    location = self._parse_gga(sentence)
                    if location:
                        return location
        except Exception as exc:  # pragma: no cover
            logger.warning("GPS read failed: %s", exc)
        return None

    def close(self) -> None:
        if self._port is not None:
            try:
                self._port.close()
            except Exception:
                pass
            self._port = None
            logger.debug("GPS serial port closed.")

    def _parse_rmc(self, sentence: str) -> Optional[Dict[str, Any]]:
        parts = sentence.split(",")
        if len(parts) < 12 or parts[2] != "A":
            return None

        latitude = _parse_latlon(parts[3], parts[4])
        longitude = _parse_latlon(parts[5], parts[6])
        speed_knots = float(parts[7]) if parts[7] else 0.0
        track_angle = float(parts[8]) if parts[8] else 0.0

        if latitude is None or longitude is None:
            return None

        return {
            "latitude": latitude,
            "longitude": longitude,
            "speed_knots": speed_knots,
            "track_angle_deg": track_angle,
            "timestamp": parts[1],
        }

    def _parse_gga(self, sentence: str) -> Optional[Dict[str, Any]]:
        parts = sentence.split(",")
        if len(parts) < 10 or parts[6] == "0":
            return None

        latitude = _parse_latlon(parts[2], parts[3])
        longitude = _parse_latlon(parts[4], parts[5])
        altitude = float(parts[9]) if parts[9] else 0.0

        if latitude is None or longitude is None:
            return None

        return {
            "latitude": latitude,
            "longitude": longitude,
            "altitude_m": altitude,
            "timestamp": parts[1],
        }
