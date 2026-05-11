"""
hardware/compass.py

Compass / magnetometer support for I2C-based modules such as HMC5883L.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from config import COMPASS_CFG

logger = logging.getLogger(__name__)

try:
    from smbus2 import SMBus  # type: ignore[import]
except ImportError:  # pragma: no cover
    SMBus = None


def _to_signed(value: int) -> int:
    return value - 0x10000 if value & 0x8000 else value


class Compass:
    """Read heading from an I2C magnetometer sensor."""

    def __init__(self) -> None:
        self.bus_id = COMPASS_CFG.I2C_BUS
        self.address = COMPASS_CFG.I2C_ADDRESS
        self._bus: Optional[SMBus] = None

    def _ensure_bus(self) -> bool:
        if SMBus is None:
            logger.warning("smbus2 not installed; compass support disabled.")
            return False
        if self._bus is not None:
            return True
        try:
            self._bus = SMBus(self.bus_id)
            logger.debug("Compass I2C bus opened: /dev/i2c-%d", self.bus_id)
            return True
        except Exception as exc:  # pragma: no cover
            logger.warning("Unable to open I2C bus %d: %s", self.bus_id, exc)
            self._bus = None
            return False

    def read_raw(self) -> Optional[tuple[int, int, int]]:
        if not self._ensure_bus():
            return None
        try:
            data = self._bus.read_i2c_block_data(self.address, 0x03, 6)
            x = _to_signed((data[0] << 8) | data[1])
            z = _to_signed((data[2] << 8) | data[3])
            y = _to_signed((data[4] << 8) | data[5])
            return x, y, z
        except Exception as exc:  # pragma: no cover
            logger.warning("Compass read failed: %s", exc)
            return None

    def read_heading(self) -> Optional[float]:
        raw = self.read_raw()
        if raw is None:
            return None
        x, y, _ = raw
        heading = math.degrees(math.atan2(y, x))
        if heading < 0:
            heading += 360.0
        heading = round(heading, 1)
        logger.debug("Compass heading read: %.1f°", heading)
        return heading

    def close(self) -> None:
        if self._bus is not None:
            try:
                self._bus.close()
            except Exception:
                pass
            self._bus = None
