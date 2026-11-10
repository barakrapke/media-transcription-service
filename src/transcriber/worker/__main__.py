"""Entrypoint: ``python -m transcriber.worker``."""

import asyncio

from transcriber.core.config import get_settings
from transcriber.core.logging import configure_logging
from transcriber.worker.main import run_worker


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    asyncio.run(run_worker(settings))


if __name__ == "__main__":
    main()
