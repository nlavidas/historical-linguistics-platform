#!/usr/bin/env python3
"""Railway 24/7 worker entrypoint for the diachronic multilingual corpus.

This script is designed to run indefinitely on Railway as a worker service.
It repeatedly runs collection cycles and sleeps between them, so the process
never exits in normal operation and the corpus keeps growing 24/7.
"""

import time
import logging
import traceback

from DIACHRONIC_MULTILINGUAL_COLLECTOR import (
    DiachronicMultilingualCollector,
    logger as base_logger,
)

logger = logging.getLogger("railway_worker_247")


def main() -> None:
    logger.info("=" * 80)
    logger.info("RAILWAY 24/7 WORKER STARTING")
    logger.info("=" * 80)

    collector = DiachronicMultilingualCollector()

    # Run indefinitely. Railway will restart the container if it ever crashes,
    # but under normal conditions this loop should keep running forever.
    while True:
        try:
            collector.run_cycle()
        except Exception as e:  # noqa: BLE001
            logger.error(f"Unexpected error in collection cycle: {e}")
            logger.error(traceback.format_exc())
            # Short cooldown after an error
            time.sleep(60)

        # Be polite to source websites: pause between cycles.
        logger.info("Railway worker sleeping for 20 minutes before next cycle...")
        time.sleep(20 * 60)


if __name__ == "__main__":
    main()
