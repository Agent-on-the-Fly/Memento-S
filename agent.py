"""Memento-S agent entry point."""


def main() -> None:
    from cli.main import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()
