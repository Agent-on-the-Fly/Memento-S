

def main() -> None:
    from tui.interaction_logger import setup_interaction_logging
    from tui.command_line import app

    setup_interaction_logging().event("package_entrypoint", module="tui.__init__")
    app()
