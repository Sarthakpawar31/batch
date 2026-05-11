"""
control/planner.py

Autonomous Decision Planner – the brain of the rover.

Implements the  OBSERVE → ANALYZE → DECIDE → ACT  loop.

This module:
  1. Polls sensors (ultrasonic, IR) at SENSOR_HZ
  2. Polls camera / leaf detector at VISION_HZ
  3. Drives the state machine based on observations
  4. Issues motor commands via JSON action messages
  5. Triggers AI inference and API calls at appropriate states
  6. Persists results to the database

IMPORTANT: The planner never manipulates GPIO directly.
  All motor commands go through:  JSON action → MotorController

Architecture:
  asyncio tasks run concurrently on a single thread:
    • _sensor_loop   – fast loop, polls distance
    • _vision_loop   – medium loop, detects leaves
    • _decision_loop – slow loop, drives FSM + actions
    • _battery_loop  – very slow loop, battery watchdog
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import cv2

from config import LOOP_CFG, MOTOR_CFG, REPORT_DIR, AI_CFG
from control.state_machine import State, StateMachine
from hardware.compass import Compass
from hardware.dht11 import DHT11Reader
from hardware.gps import GPSReader
from hardware.motors import MotorController
from hardware.ultrasonic import async_read_distance_cm, read_ir_sensors
from vision.camera import Camera
from vision.leaf_detection import best_leaf
from ai.classify import DiseaseClassifier, PredictionResult
from ai.openrouter_client import OpenRouterClient
from memory.database import RoverDatabase

logger = logging.getLogger(__name__)


# ── Shared sensor state (written by sensor_loop, read by decision_loop) ──────
class _SensorData:
    distance_cm: Optional[float] = None
    obstacle:    bool            = False
    ir_left:     bool            = False
    ir_right:    bool            = False
    leaf_found:  bool            = False
    last_frame:  Any             = None   # numpy array | None
    temperature_c: Optional[float] = None
    humidity_pct: Optional[float] = None
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    gps_altitude_m: Optional[float] = None
    heading_deg: Optional[float] = None


_sensors = _SensorData()


class Planner:
    """
    Orchestrates all rover subsystems.

    Usage::

        planner = Planner()
        await planner.run()      # blocks until SIGINT or error
    """

    def __init__(self) -> None:
        self.sm           = StateMachine()
        self.motors       = MotorController()
        self.camera       = Camera()
        self.classifier   = DiseaseClassifier.get()
        self.api          = OpenRouterClient()
        self.db           = RoverDatabase()
        self.dht_sensor   = DHT11Reader()
        self.gps_reader   = GPSReader()
        self.compass      = Compass()
        self._running     = False

        # Register state-change logging
        self.sm.on_transition(self._on_state_change)

    # ── Entry point ───────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Start all asyncio tasks and run until stopped."""
        await self.db.init()
        session_id = await self.db.start_session()
        logger.info("Rover starting – session %d", session_id)

        self._running = True
        self.sm.transition(State.EXPLORING)

        tasks = [
            asyncio.create_task(self._sensor_loop(),      name="sensor"),
            asyncio.create_task(self._environment_loop(), name="environment"),
            asyncio.create_task(self._vision_loop(),      name="vision"),
            asyncio.create_task(self._decision_loop(),    name="decision"),
            asyncio.create_task(self._battery_loop(),     name="battery"),
        ]

        if await self._wait_for_clear_path(timeout=1.0):
            self.motors.forward()
        else:
            logger.warning(
                "Starting path not clear; entering obstacle state instead of driving."
            )
            self.sm.transition(State.OBSTACLE)

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Rover stopping…")
        finally:
            await self._shutdown(session_id)

    async def stop(self) -> None:
        self._running = False

    # ── Sensor loop (fast) ────────────────────────────────────────────────────

    async def _sensor_loop(self) -> None:
        interval = 1.0 / LOOP_CFG.SENSOR_HZ
        while self._running:
            dist = await async_read_distance_cm()
            ir_left, ir_right = await asyncio.get_event_loop().run_in_executor(
                None, read_ir_sensors
            )

            _sensors.distance_cm = dist
            _sensors.ir_left     = ir_left
            _sensors.ir_right    = ir_right
            _sensors.obstacle    = (
                ir_left or ir_right
                or (dist is not None and dist < MOTOR_CFG.OBSTACLE_DIST_CM)
            )
            await asyncio.sleep(interval)

    # ── Vision loop (medium) ──────────────────────────────────────────────────

    async def _vision_loop(self) -> None:
        interval = 1.0 / LOOP_CFG.VISION_HZ
        loop     = asyncio.get_event_loop()

        while self._running:
            # Only run detection while exploring to save CPU
            if self.sm.is_in(State.EXPLORING):
                frame = await self.camera.async_get_frame()
                if frame is not None:
                    _sensors.last_frame = frame
                    leaf = await loop.run_in_executor(
                        None, best_leaf, frame
                    )
                    _sensors.leaf_found = leaf is not None
            await asyncio.sleep(interval)

    async def _environment_loop(self) -> None:
        interval = 1.0 / LOOP_CFG.ENV_HZ
        loop = asyncio.get_event_loop()

        while self._running:
            temperature, humidity = await loop.run_in_executor(
                None, self.dht_sensor.read
            )
            heading = await loop.run_in_executor(
                None, self.compass.read_heading
            )
            location = await loop.run_in_executor(
                None, self.gps_reader.read_location
            )

            _sensors.temperature_c = temperature
            _sensors.humidity_pct = humidity
            _sensors.heading_deg = heading

            if location is not None:
                _sensors.gps_latitude = location.get("latitude")
                _sensors.gps_longitude = location.get("longitude")
                _sensors.gps_altitude_m = location.get("altitude_m")

            await asyncio.sleep(interval)

    # ── Decision loop (slow) ──────────────────────────────────────────────────

    async def _decision_loop(self) -> None:
        interval = 1.0 / LOOP_CFG.DECISION_HZ

        while self._running:
            state = self.sm.state

            if state == State.EXPLORING:
                await self._handle_exploring()

            elif state == State.OBSTACLE:
                await self._handle_obstacle()

            elif state == State.PLANT_FOUND:
                await self._handle_plant_found()

            elif state == State.INSPECTING:
                await self._handle_inspecting()

            elif state == State.REPORTING:
                # reporting is triggered from inspecting; wait here
                pass

            elif state == State.RETURNING:
                await self._handle_returning()

            await asyncio.sleep(interval)

    # ── Battery watchdog (very slow) ──────────────────────────────────────────

    async def _battery_loop(self) -> None:
        while self._running:
            await asyncio.sleep(LOOP_CFG.BATTERY_CHECK_S)
            # TODO: read ADC pin for voltage; for now simulate OK
            battery_ok = True
            if not battery_ok:
                logger.warning("Battery low – returning to base.")
                self.motors.stop()
                self.sm.transition(State.RETURNING)

    # ── State handlers ────────────────────────────────────────────────────────

    async def _handle_exploring(self) -> None:
        if _sensors.obstacle:
            logger.info("Obstacle at %.1f cm", _sensors.distance_cm or 0)
            self.motors.stop()
            self.sm.transition(State.OBSTACLE)

        elif _sensors.leaf_found:
            logger.info("Plant detected – approaching.")
            self.motors.stop()
            self.sm.transition(State.PLANT_FOUND)

        else:
            # Keep moving forward
            if self.motors.current_speed == 0:
                self.motors.forward()

    async def _wait_for_clear_path(self, timeout: float = 2.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            dist = await async_read_distance_cm()
            ir_left, ir_right = await asyncio.get_event_loop().run_in_executor(
                None, read_ir_sensors
            )
            _sensors.distance_cm = dist
            _sensors.ir_left     = ir_left
            _sensors.ir_right    = ir_right

            if (
                not ir_left
                and not ir_right
                and dist is not None
                and dist >= MOTOR_CFG.CLEAR_DIST_CM
            ):
                return True
            await asyncio.sleep(0.1)
        return False

    async def _handle_obstacle(self) -> None:
        # Avoidance runs synchronously in thread executor to avoid blocking
        loop = asyncio.get_event_loop()
        max_attempts = 3

        for attempt in range(1, max_attempts + 1):
            logger.info("Obstacle avoidance attempt %d/%d", attempt, max_attempts)
            await loop.run_in_executor(None, self.motors.avoid_obstacle)
            logger.debug("Checking path clearance after avoidance.")
            if await self._wait_for_clear_path(timeout=1.5):
                logger.info("Path clear after avoidance.")
                self.sm.transition(State.EXPLORING)
                self.motors.forward()
                return

            logger.warning(
                "Path still blocked at %.1f cm after avoidance.",
                _sensors.distance_cm or 0.0,
            )

        logger.error(
            "Unable to clear obstacle after %d attempts. Stopping and retrying.",
            max_attempts,
        )
        self.motors.stop()
        # Keep state OBSTACLE so the decision loop will retry avoidance on the next tick.

    async def _handle_plant_found(self) -> None:
        # Brief pause to stabilise camera
        await asyncio.sleep(0.3)
        self.sm.transition(State.INSPECTING)

    async def _handle_inspecting(self) -> None:
        """
        Full inspection pipeline:
          capture → detect leaf → classify → store → API
        """
        # 1. Capture high-quality still
        loop  = asyncio.get_event_loop()
        frame = await loop.run_in_executor(None, self.camera.capture_still)

        if frame is None:
            logger.warning("Failed to capture frame; resuming exploration.")
            self.sm.transition(State.EXPLORING)
            self.motors.forward()
            return

        # 2. Detect best leaf region
        leaf = await loop.run_in_executor(None, best_leaf, frame)
        if leaf is None:
            logger.info("No valid leaf found in still; resuming.")
            self.sm.transition(State.EXPLORING)
            self.motors.forward()
            return

        # 3. Classify disease
        result: Optional[PredictionResult] = await loop.run_in_executor(
            None, self.classifier.predict, leaf.roi
        )

        if result is None:
            logger.warning("Classifier returned None; resuming.")
            self.sm.transition(State.EXPLORING)
            self.motors.forward()
            return

        logger.info(
            "Classification: %s  conf=%.2f  severity=%s",
            result.disease, result.confidence, result.severity
        )

        # 4. Save annotated image
        img_path = ""
        if not result.is_healthy:
            img_path = await self._save_image(frame, result)

        # 5. Get treatment advice (async, may call OpenRouter)
        self.sm.transition(State.REPORTING)
        treatment = await self.api.get_treatment_advice(result)

        # 6. Persist to database
        await self.db.save_detection(
            disease     = result.disease,
            confidence  = result.confidence,
            severity    = result.severity,
            image_path  = img_path,
            treatment   = treatment,
        )

        # 7. Optionally speak result (TTS stub)
        self._speak(result, treatment)

        # 8. Resume exploration
        self.sm.transition(State.EXPLORING)
        self.motors.forward()

    async def _handle_returning(self) -> None:
        """Drive backwards to the charging area (simple reverse)."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.motors.backward)
        await asyncio.sleep(3)  # reverse 3s
        await loop.run_in_executor(None, self.motors.stop)
        self.sm.transition(State.IDLE)
        self._running = False

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _save_image(
        self, frame: Any, result: PredictionResult
    ) -> str:
        """Save annotated frame to reports/ and return relative path."""
        ts       = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{ts}_{result.disease.replace(' ', '_')}.jpg"
        path     = REPORT_DIR / filename

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, lambda: cv2.imwrite(str(path), frame)
        )
        return str(path)

    def _speak(self, result: PredictionResult, treatment: Dict[str, Any]) -> None:
        """
        Text-to-speech announcement (optional).
        Guarded by try/except so missing pyttsx3 doesn't crash the rover.
        """
        try:
            import pyttsx3                          # type: ignore[import]
            engine = pyttsx3.init()
            msg = (
                f"Plant detected. {result.disease}. "
                f"Confidence {int(result.confidence * 100)} percent. "
                f"Severity: {result.severity}."
            )
            engine.say(msg)
            engine.runAndWait()
        except Exception:
            pass   # TTS is optional

    def _on_state_change(self, old: State, new: State) -> None:
        logger.debug("FSM transition logged: %s → %s", old.name, new.name)

    # ── Shutdown ──────────────────────────────────────────────────────────────

    async def _shutdown(self, session_id: int) -> None:
        self.motors.stop()
        self.motors.cleanup()
        self.gps_reader.close()
        self.camera.release()

        # End session with a summary
        detections = await self.db.get_session_detections()
        summary    = await self.api.generate_session_report(detections)
        await self.db.end_session(summary)
        await self.db.close()

        logger.info("Shutdown complete.  Session summary:\n%s", summary)
