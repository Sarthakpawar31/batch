"""
control/state_machine.py

Finite State Machine for the autonomous rover.

States
──────
IDLE          → waiting to start
EXPLORING     → moving forward, scanning for plants
OBSTACLE      → obstacle detected, executing avoidance
PLANT_FOUND   → plant/leaf region detected, approaching
INSPECTING    → stopped, capturing image, running AI
REPORTING     → sending results to API, saving to DB
RETURNING     → battery low, returning to charging base
ERROR         → unrecoverable hardware fault

Allowed transitions (all others are rejected):
  IDLE          → EXPLORING
  EXPLORING     → OBSTACLE | PLANT_FOUND | RETURNING | ERROR
  OBSTACLE      → EXPLORING
  PLANT_FOUND   → INSPECTING
  INSPECTING    → REPORTING | EXPLORING
  REPORTING     → EXPLORING
  RETURNING     → IDLE
  ERROR         → IDLE  (manual reset only)
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Callable, Dict, Optional, Set

logger = logging.getLogger(__name__)


class State(Enum):
    IDLE        = auto()
    EXPLORING   = auto()
    OBSTACLE    = auto()
    PLANT_FOUND = auto()
    INSPECTING  = auto()
    REPORTING   = auto()
    RETURNING   = auto()
    ERROR       = auto()


# Valid transitions: {from_state: {allowed_to_states}}
_TRANSITIONS: Dict[State, Set[State]] = {
    State.IDLE:        {State.EXPLORING},
    State.EXPLORING:   {State.OBSTACLE, State.PLANT_FOUND,
                        State.RETURNING, State.ERROR},
    State.OBSTACLE:    {State.EXPLORING},
    State.PLANT_FOUND: {State.INSPECTING},
    State.INSPECTING:  {State.REPORTING, State.EXPLORING},
    State.REPORTING:   {State.EXPLORING},
    State.RETURNING:   {State.IDLE},
    State.ERROR:       {State.IDLE},
}

# Callbacks type alias
StateCallback = Callable[[State, State], None]


class StateMachine:
    """
    Lightweight FSM with optional transition callbacks.

    Usage::

        sm = StateMachine()
        sm.on_transition(my_callback)   # optional
        sm.transition(State.EXPLORING)
        print(sm.state)                 # State.EXPLORING
    """

    def __init__(self) -> None:
        self._state:     State                    = State.IDLE
        self._callbacks: list[StateCallback]      = []
        self._history:   list[tuple[State, State]] = []

    # ── State access ──────────────────────────────────────────────────────────

    @property
    def state(self) -> State:
        return self._state

    def is_in(self, *states: State) -> bool:
        return self._state in states

    # ── Transitions ───────────────────────────────────────────────────────────

    def transition(self, new_state: State) -> bool:
        """
        Attempt a state transition.

        Returns True if the transition was accepted, False if illegal.
        """
        allowed = _TRANSITIONS.get(self._state, set())
        if new_state not in allowed:
            logger.warning(
                "Illegal transition %s → %s  (allowed: %s)",
                self._state.name,
                new_state.name,
                {s.name for s in allowed},
            )
            return False

        old_state = self._state
        self._state = new_state
        self._history.append((old_state, new_state))

        logger.info("State: %s → %s", old_state.name, new_state.name)

        for cb in self._callbacks:
            try:
                cb(old_state, new_state)
            except Exception as exc:
                logger.error("State callback error: %s", exc)

        return True

    def force_state(self, new_state: State) -> None:
        """
        Force-set state without validation.
        Use ONLY for error recovery / testing.
        """
        old = self._state
        self._state = new_state
        self._history.append((old, new_state))
        logger.warning("Force state: %s → %s", old.name, new_state.name)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def on_transition(self, callback: StateCallback) -> None:
        """Register a callback invoked on every successful transition."""
        self._callbacks.append(callback)

    # ── Debug / introspection ─────────────────────────────────────────────────

    @property
    def history(self) -> list:
        return list(self._history)

    def __repr__(self) -> str:
        return f"<StateMachine state={self._state.name}>"
