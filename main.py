"""
main.py

Agricultural Rover – Entry Point
=================================

Boots all subsystems, registers signal handlers for graceful shutdown,
and runs the autonomous event loop.

Usage:
    python main.py

Environment variables (can be in .env):
    OPENROUTER_API_KEY   – your OpenRouter API key
    LOG_LEVEL            – DEBUG | INFO | WARNING  (default: INFO)
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass   # uvloop is optional; stdlib asyncio is fine

from config import LOG_LEVEL, LOG_FORMAT
from hardware.gpio_config import init_gpio, cleanup_gpio
from control.planner import Planner


# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = getattr(logging, LOG_LEVEL, logging.INFO),
    format  = LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("rover.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    """Initialise hardware and run the autonomous planner."""
    logger.info("═══════════════════════════════════════")
    logger.info("   AgriRover starting up               ")
    logger.info("═══════════════════════════════════════")

    init_gpio()

    planner = Planner()

    # Graceful SIGINT / SIGTERM handler
    loop = asyncio.get_running_loop()

    def _handle_signal(sig: signal.Signals) -> None:
        logger.info("Signal %s received – stopping rover.", sig.name)
        asyncio.create_task(planner.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: _handle_signal(s))

    try:
        await planner.run()
    except Exception as exc:
        logger.critical("Unhandled exception: %s", exc, exc_info=True)
    finally:
        cleanup_gpio()
        logger.info("Rover shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
