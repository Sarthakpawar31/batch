"""
hardware/motors.py

Low-level motor control via L298N dual H-bridge.

┌─────────────────────────────────────────────────────────────────────┐
│  L298N wiring summary                                               │
│  ENA  ──►  Left  motor PWM speed  (GPIO 12)                        │
│  IN1  ──►  Left  motor direction A (GPIO 20)                       │
│  IN2  ──►  Left  motor direction B (GPIO 21)                       │
│  ENB  ──►  Right motor PWM speed  (GPIO 13)                        │
│  IN3  ──►  Right motor direction A (GPIO 19)                       │
│  IN4  ──►  Right motor direction B (GPIO 26)                       │
└─────────────────────────────────────────────────────────────────────┘

The MotorController class is the ONLY place in the project where
GPIO output is written.  All higher layers (planner, state machine)
call the public methods below.
"""

from __future__ import annotations

import logging
import time

from config import MOTOR_PINS, MOTOR_CFG
from hardware.gpio_config import get_gpio

logger = logging.getLogger(__name__)

GPIO = get_gpio()


class MotorController:
    """
    Manages both DC motors connected through an L298N.

    Usage::

        mc = MotorController()
        mc.forward(speed=70)
        time.sleep(1)
        mc.stop()
        mc.cleanup()
    """

    def __init__(self) -> None:
        self._pwm_left:  GPIO.PWM | None = None   # type: ignore[name-defined]
        self._pwm_right: GPIO.PWM | None = None   # type: ignore[name-defined]
        self._current_speed: int = 0
        self._setup()

    # ── Setup / teardown ──────────────────────────────────────────────────────

    def _setup(self) -> None:
        self._pwm_left  = GPIO.PWM(MOTOR_PINS.ENA, MOTOR_CFG.PWM_FREQ)
        self._pwm_right = GPIO.PWM(MOTOR_PINS.ENB, MOTOR_CFG.PWM_FREQ)
        self._pwm_left.start(0)
        self._pwm_right.start(0)
        logger.debug("MotorController ready.")

    def cleanup(self) -> None:
        self.stop()
        if self._pwm_left:
            self._pwm_left.stop()
        if self._pwm_right:
            self._pwm_right.stop()
        logger.debug("MotorController cleaned up.")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _set_left(self, fwd: bool, speed: int) -> None:
        """Drive left motor.  fwd=True → forward direction."""
        GPIO.output(MOTOR_PINS.IN1, GPIO.HIGH if fwd else GPIO.LOW)
        GPIO.output(MOTOR_PINS.IN2, GPIO.LOW  if fwd else GPIO.HIGH)
        self._pwm_left.ChangeDutyCycle(speed)   # type: ignore[union-attr]

    def _set_right(self, fwd: bool, speed: int) -> None:
        """Drive right motor.  fwd=True → forward direction."""
        GPIO.output(MOTOR_PINS.IN3, GPIO.HIGH if fwd else GPIO.LOW)
        GPIO.output(MOTOR_PINS.IN4, GPIO.LOW  if fwd else GPIO.HIGH)
        self._pwm_right.ChangeDutyCycle(speed)  # type: ignore[union-attr]

    # ── Public movement API ───────────────────────────────────────────────────

    def forward(self, speed: int | None = None) -> None:
        s = speed if speed is not None else MOTOR_CFG.DEFAULT_SPEED
        self._set_left(True, s)
        self._set_right(True, s)
        self._current_speed = s
        logger.debug("forward speed=%d", s)

    def backward(self, speed: int | None = None) -> None:
        s = speed if speed is not None else MOTOR_CFG.DEFAULT_SPEED
        self._set_left(False, s)
        self._set_right(False, s)
        self._current_speed = s
        logger.debug("backward speed=%d", s)

    def turn_left(self, speed: int | None = None) -> None:
        """Pivot left: left motor backward, right motor forward."""
        s = speed if speed is not None else MOTOR_CFG.TURN_SPEED
        self._set_left(False, s)
        self._set_right(True, s)
        self._current_speed = s
        logger.debug("turn_left speed=%d", s)

    def turn_right(self, speed: int | None = None) -> None:
        """Pivot right: left motor forward, right motor backward."""
        s = speed if speed is not None else MOTOR_CFG.TURN_SPEED
        self._set_left(True, s)
        self._set_right(False, s)
        self._current_speed = s
        logger.debug("turn_right speed=%d", s)

    def stop(self) -> None:
        GPIO.output(MOTOR_PINS.IN1, GPIO.LOW)
        GPIO.output(MOTOR_PINS.IN2, GPIO.LOW)
        GPIO.output(MOTOR_PINS.IN3, GPIO.LOW)
        GPIO.output(MOTOR_PINS.IN4, GPIO.LOW)
        self._pwm_left.ChangeDutyCycle(0)    # type: ignore[union-attr]
        self._pwm_right.ChangeDutyCycle(0)   # type: ignore[union-attr]
        self._current_speed = 0
        logger.debug("stop")

    # ── Convenience manoeuvres ────────────────────────────────────────────────

    def avoid_obstacle(self) -> None:
        """
        Simple avoidance sequence:
          reverse → pause → turn right → resume forward
        This runs synchronously so callers must be prepared to block briefly.
        """
        logger.info("Obstacle avoidance sequence start.")
        self.backward()
        time.sleep(MOTOR_CFG.REVERSE_SECS)
        self.turn_right()
        time.sleep(MOTOR_CFG.TURN_SECS)
        self.stop()
        logger.info("Obstacle avoidance sequence complete.")

    @property
    def current_speed(self) -> int:
        return self._current_speed
