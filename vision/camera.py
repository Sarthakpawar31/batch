from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import cv2
import numpy as np

from config import CAM_CFG

logger = logging.getLogger(__name__)


class CameraError(RuntimeError):
    pass


class _PiCameraBackend:
    def __init__(self) -> None:
        from picamera2 import Picamera2
        self._cam = Picamera2()
        cfg = self._cam.create_video_configuration(
            main={"size": (CAM_CFG.WIDTH, CAM_CFG.HEIGHT), "format": "BGR888"}
        )
        self._cam.configure(cfg)
        self._cam.start()
        time.sleep(2)  # let auto-exposure settle
        logger.info("PiCamera2 opened at %dx%d", CAM_CFG.WIDTH, CAM_CFG.HEIGHT)

    def read(self) -> Optional[np.ndarray]:
        try:
            return self._cam.capture_array()
        except Exception as e:
            logger.error("PiCamera2 read error: %s", e)
            return None

    def release(self) -> None:
        self._cam.stop()
        logger.info("PiCamera2 released.")


class _OpenCVBackend:
    def __init__(self, index: int = 0) -> None:
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise CameraError(f"Cannot open camera index {index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_CFG.WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_CFG.HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS,          CAM_CFG.FRAMERATE)
        time.sleep(2)
        for _ in range(5):
            self._cap.read()
        logger.info("OpenCV camera %d ready.", index)

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        self._cap.release()
        logger.info("OpenCV camera released.")


class Camera:
    def __init__(self) -> None:
        self._backend: _PiCameraBackend | _OpenCVBackend
        if CAM_CFG.USE_PICAMERA:
            try:
                self._backend = _PiCameraBackend()
            except Exception as exc:
                logger.warning("PiCamera2 failed (%s), falling back to OpenCV.", exc)
                self._backend = _OpenCVBackend()
        else:
            self._backend = _OpenCVBackend()
        self._frame_count = 0
        self._last_frame = None

    def get_frame(self) -> Optional[np.ndarray]:
        self._frame_count += 1
        if self._frame_count % CAM_CFG.FRAME_SKIP == 0:
            self._last_frame = self._backend.read()
        return self._last_frame

    async def async_get_frame(self) -> Optional[np.ndarray]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_frame)

    def capture_still(self) -> Optional[np.ndarray]:
        return self._backend.read()

    def release(self) -> None:
        self._backend.release()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *args: object) -> None:
        self.release()
