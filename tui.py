#!/usr/bin/env python3

from core.config.logging import setup_logging
from tui.command_line import app

if __name__ == "__main__":
    setup_logging(log_file="memento.log", console_output=False)
    app()
