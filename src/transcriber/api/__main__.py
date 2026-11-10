"""Entrypoint: ``python -m transcriber.api``."""

import uvicorn

from transcriber.core.config import get_settings
from transcriber.core.logging import configure_logging


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    uvicorn.run(
        "transcriber.api.app:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        log_config=None,
        proxy_headers=True,
        timeout_graceful_shutdown=10,
    )


if __name__ == "__main__":
    main()
