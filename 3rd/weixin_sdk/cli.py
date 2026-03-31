"""
Weixin SDK CLI - Command-line interface for managing Weixin bot accounts.

This module provides a CLI tool for managing Weixin accounts, including
listing, adding, removing, enabling, disabling, and testing accounts.

Usage:
    weixin list                           # List all accounts
    weixin add bot1 --token abc123        # Add a new account
    weixin remove bot1                    # Remove an account
    weixin enable bot1                    # Enable an account
    weixin disable bot1                   # Disable an account
    weixin show bot1                      # Show account details
    weixin test bot1                      # Test account configuration
    weixin upload                         # Upload logs for troubleshooting
    weixin upload --account bot1          # Upload logs for specific account
    weixin upload --date 2026-03-25       # Upload logs from specific date
    weixin upload --all                   # Upload all logs

Environment Variables:
    WEIXIN_SDK_STATE_DIR: Override default state directory (~/.weixin_sdk)
    WEIXIN_LOG_UPLOAD_URL: URL for uploading logs
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

from .account import WeixinAccountManager, WeixinAccount
from .exceptions import WeixinError
from .log_upload import (
    LogUploader,
    LogUploadError,
    get_upload_url,
    format_size,
    print_progress,
)

# Version info
__version__ = "1.0.0"

# Default state directory
DEFAULT_STATE_DIR = Path.home() / ".weixin_sdk"

# Environment variable for state directory
STATE_DIR_ENV = "WEIXIN_SDK_STATE_DIR"


def get_state_dir() -> Path:
    """Get the state directory from environment or default."""
    if STATE_DIR_ENV in os.environ:
        return Path(os.environ[STATE_DIR_ENV])
    return DEFAULT_STATE_DIR


def validate_token(token: str) -> bool:
    """
    Validate token format (basic format check).

    The Weixin API token should be a non-empty string.
    For more robust validation, we check:
    - Not empty
    - Minimum length (32 characters for typical tokens)
    - Alphanumeric with allowed special chars

    Args:
        token: Token string to validate

    Returns:
        True if valid, False otherwise
    """
    if not token or not token.strip():
        return False

    # Check minimum length (typical tokens are at least 32 chars)
    if len(token.strip()) < 10:
        return False

    # Check for valid characters (alphanumeric, hyphen, underscore)
    if not re.match(r"^[a-zA-Z0-9_\-]+$", token.strip()):
        return False

    return True


def format_json(data: Any, indent: int = 2) -> str:
    """Format data as JSON."""
    return json.dumps(data, indent=indent, ensure_ascii=False, default=str)


def format_table(headers: List[str], rows: List[List[str]]) -> str:
    """
    Format data as a simple table (fallback when tabulate not available).

    Args:
        headers: Column headers
        rows: Data rows

    Returns:
        Formatted table string
    """
    if not rows:
        return "No data"

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    # Add padding
    col_widths = [w + 2 for w in col_widths]

    # Format header
    header_line = "".join(h.ljust(w) for h, w in zip(headers, col_widths))
    separator = "=" * len(header_line)

    # Format rows
    formatted_rows = []
    for row in rows:
        row_line = "".join(str(cell).ljust(w) for cell, w in zip(row, col_widths))
        formatted_rows.append(row_line)

    return "\n".join([separator, header_line, separator] + formatted_rows + [""])


def try_tabulate(headers: List[str], rows: List[List[str]]) -> Optional[str]:
    """
    Try to format using tabulate library if available.

    Args:
        headers: Column headers
        rows: Data rows

    Returns:
        Formatted table or None if tabulate not available
    """
    try:
        from tabulate import tabulate

        return tabulate(rows, headers=headers, tablefmt="grid")
    except ImportError:
        return None


class OutputFormat:
    """Output format options."""

    JSON = "json"
    TABLE = "table"


# -----------------------------------------------------------------------------
# Command Implementations
# -----------------------------------------------------------------------------


def cmd_list(manager: WeixinAccountManager, args: argparse.Namespace) -> int:
    """
    List all registered accounts.

    Args:
        manager: Account manager instance
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success)
    """
    accounts = manager.list_accounts()

    if not accounts:
        print("No accounts registered.")
        return 0

    if args.output == OutputFormat.JSON:
        # List all account details in JSON format
        result = {}
        for account_id in accounts:
            account = manager.get_account(account_id)
            if account:
                result[account_id] = account.to_dict()
        print(format_json(result))
    else:
        # Table format
        rows = []
        for account_id in accounts:
            account = manager.get_account(account_id)
            if account:
                rows.append(
                    [
                        account_id,
                        account.name or "-",
                        "Yes" if account.enabled else "No",
                        "Yes" if account.configured else "No",
                        account.base_url,
                    ]
                )

        headers = ["Account ID", "Name", "Enabled", "Configured", "Base URL"]

        table_output = try_tabulate(headers, rows)
        if table_output:
            print(table_output)
        else:
            print(format_table(headers, rows))

    return 0


def cmd_add(manager: WeixinAccountManager, args: argparse.Namespace) -> int:
    """
    Add a new account.

    Args:
        manager: Account manager instance
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    account_id = args.account_id

    # Check if account already exists
    existing = manager.get_account(account_id)
    if existing:
        print(f"Error: Account '{account_id}' already exists.", file=sys.stderr)
        print(
            f"Use 'weixin update {account_id}' to modify existing accounts.",
            file=sys.stderr,
        )
        return 1

    # Validate token if provided
    if args.token and not validate_token(args.token):
        print(f"Error: Invalid token format.", file=sys.stderr)
        print(
            f"Token must be at least 10 characters and contain only alphanumeric characters, hyphens, and underscores.",
            file=sys.stderr,
        )
        return 1

    # Create account
    account = WeixinAccount(
        account_id=account_id,
        name=args.name,
        enabled=True,
        base_url=args.base_url or "https://ilinkai.weixin.qq.com",
        token=args.token,
        configured=args.token is not None,
    )

    try:
        manager.register_account(account_id, account)
        print(f"Account '{account_id}' added successfully.")

        if args.output == OutputFormat.JSON:
            print(format_json(account.to_dict()))

        return 0
    except Exception as e:
        print(f"Error: Failed to add account: {e}", file=sys.stderr)
        return 1


def cmd_remove(manager: WeixinAccountManager, args: argparse.Namespace) -> int:
    """
    Remove an account.

    Args:
        manager: Account manager instance
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    account_id = args.account_id

    # Check if account exists
    existing = manager.get_account(account_id)
    if not existing:
        print(f"Error: Account '{account_id}' not found.", file=sys.stderr)
        return 1

    try:
        manager.unregister_account(account_id)
        print(f"Account '{account_id}' removed successfully.")
        return 0
    except Exception as e:
        print(f"Error: Failed to remove account: {e}", file=sys.stderr)
        return 1


def cmd_enable(manager: WeixinAccountManager, args: argparse.Namespace) -> int:
    """
    Enable an account.

    Args:
        manager: Account manager instance
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    account_id = args.account_id

    # Check if account exists
    existing = manager.get_account(account_id)
    if not existing:
        print(f"Error: Account '{account_id}' not found.", file=sys.stderr)
        return 1

    try:
        manager.update_account(account_id, enabled=True)
        print(f"Account '{account_id}' enabled.")
        return 0
    except Exception as e:
        print(f"Error: Failed to enable account: {e}", file=sys.stderr)
        return 1


def cmd_disable(manager: WeixinAccountManager, args: argparse.Namespace) -> int:
    """
    Disable an account.

    Args:
        manager: Account manager instance
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    account_id = args.account_id

    # Check if account exists
    existing = manager.get_account(account_id)
    if not existing:
        print(f"Error: Account '{account_id}' not found.", file=sys.stderr)
        return 1

    try:
        manager.update_account(account_id, enabled=False)
        print(f"Account '{account_id}' disabled.")
        return 0
    except Exception as e:
        print(f"Error: Failed to disable account: {e}", file=sys.stderr)
        return 1


def cmd_show(manager: WeixinAccountManager, args: argparse.Namespace) -> int:
    """
    Show account details.

    Args:
        manager: Account manager instance
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    account_id = args.account_id

    # Get account
    account = manager.get_account(account_id)
    if not account:
        print(f"Error: Account '{account_id}' not found.", file=sys.stderr)
        return 1

    if args.output == OutputFormat.JSON:
        print(format_json(account.to_dict()))
    else:
        # Human-readable format
        print(f"Account ID:    {account.account_id}")
        print(f"Name:          {account.name or '(not set)'}")
        print(f"Enabled:       {'Yes' if account.enabled else 'No'}")
        print(f"Configured:    {'Yes' if account.configured else 'No'}")
        print(f"Base URL:      {account.base_url}")
        print(f"CDN Base URL:  {account.cdn_base_url}")
        print(
            f"Token:         {'***' + account.token[-4:] if account.token else '(not set)'}"
        )
        if account.route_tag is not None:
            print(f"Route Tag:     {account.route_tag}")
        else:
            print(f"Route Tag:     (not set)")

        # Show context tokens if any
        tokens = manager.find_accounts_by_context_token(account_id)
        if tokens:
            print(f"Context Tokens: {len(tokens)} user(s)")

        print()

    return 0


def cmd_test(manager: WeixinAccountManager, args: argparse.Namespace) -> int:
    """
    Test account configuration.

    This validates the account configuration without making actual API connections.

    Args:
        manager: Account manager instance
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    account_id = args.account_id

    # Get account
    account = manager.get_account(account_id)
    if not account:
        print(f"Error: Account '{account_id}' not found.", file=sys.stderr)
        return 1

    # Test configuration
    errors = []
    warnings = []

    # Check if token is set
    if not account.token:
        errors.append("Token is not set")
    elif not validate_token(account.token):
        warnings.append("Token format may be invalid")

    # Check base URL format
    if not account.base_url:
        errors.append("Base URL is not set")
    elif not account.base_url.startswith(("http://", "https://")):
        errors.append("Base URL must start with http:// or https://")

    # Check if enabled
    if not account.enabled:
        warnings.append("Account is currently disabled")

    # Check if configured
    if not account.configured:
        warnings.append("Account is not marked as configured")

    # Output results
    print(f"Testing account: {account_id}")
    print("-" * 40)

    if errors:
        print("ERRORS:")
        for error in errors:
            print(f"  [X] {error}")
        print()

    if warnings:
        print("WARNINGS:")
        for warning in warnings:
            print(f"  [!] {warning}")
        print()

    if not errors and not warnings:
        print("[OK] Account configuration looks valid!")
        print()
        return 0
    elif not errors:
        print("[OK] No critical errors found.")
        print()
        return 0
    else:
        print("[FAIL] Configuration test failed.")
        print()
        return 1


def cmd_update(manager: WeixinAccountManager, args: argparse.Namespace) -> int:
    """
    Update account configuration.

    Args:
        manager: Account manager instance
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    account_id = args.account_id

    # Check if account exists
    existing = manager.get_account(account_id)
    if not existing:
        print(f"Error: Account '{account_id}' not found.", file=sys.stderr)
        return 1

    # Build update kwargs
    update_kwargs: Dict[str, Any] = {}

    if args.name is not None:
        update_kwargs["name"] = args.name

    if args.token is not None:
        if not validate_token(args.token):
            print(f"Error: Invalid token format.", file=sys.stderr)
            return 1
        update_kwargs["token"] = args.token
        update_kwargs["configured"] = True

    if args.base_url is not None:
        update_kwargs["base_url"] = args.base_url

    if args.no_token:
        update_kwargs["token"] = None
        update_kwargs["configured"] = False

    try:
        if update_kwargs:
            manager.update_account(account_id, **update_kwargs)
            print(f"Account '{account_id}' updated successfully.")
        else:
            print(f"No changes to account '{account_id}'.")

        if args.output == OutputFormat.JSON:
            account = manager.get_account(account_id)
            if account:
                print(format_json(account.to_dict()))

        return 0
    except Exception as e:
        print(f"Error: Failed to update account: {e}", file=sys.stderr)
        return 1


def cmd_upload(manager: WeixinAccountManager, args: argparse.Namespace) -> int:
    """
    Upload logs for troubleshooting.

    Collects logs from ~/.weixin_sdk/logs/, compresses them into a ZIP archive,
    and uploads them to the configured endpoint.

    Args:
        manager: Account manager instance
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Resolve upload URL
    state_dir = args.state_dir or get_state_dir()
    upload_url = get_upload_url(
        cli_url=args.url,
        state_dir=state_dir,
    )

    if not upload_url:
        print("Error: No upload URL configured.", file=sys.stderr)
        print("\nPlease provide an upload URL via one of:", file=sys.stderr)
        print(f"  1. Command line: --url <url>", file=sys.stderr)
        print(f"  2. Environment variable: WEIXIN_LOG_UPLOAD_URL", file=sys.stderr)
        print(
            f"  3. Config file: {state_dir / 'openclaw.json'} with 'logUploadUrl' field",
            file=sys.stderr,
        )
        return 1

    # Initialize uploader
    uploader = LogUploader(state_dir=state_dir)

    try:
        # Collect logs
        if args.debug:
            print(f"Collecting logs from: {uploader.log_dir}")

        log_files = uploader.collect_logs(
            account_id=args.account,
            date=args.date,
            all_logs=args.all,
        )

        if not log_files:
            print("No log files found matching the specified criteria.")
            return 0

        if args.debug:
            print(f"Found {len(log_files)} log file(s):")
            for f in log_files:
                size = f.stat().st_size
                print(f"  - {f.name} ({format_size(size)})")

        # Show summary
        total_size = sum(f.stat().st_size for f in log_files)
        print(f"Uploading {len(log_files)} log file(s) ({format_size(total_size)})...")

        # Upload logs
        result = uploader.upload_logs(
            upload_url=upload_url,
            log_files=log_files,
            progress_callback=print_progress if not args.debug else None,
            remove_archive=not args.keep_archive,
        )

        if args.output == OutputFormat.JSON:
            print(format_json(result))
        else:
            print(f"\nUpload successful!")
            if "response" in result and isinstance(result["response"], dict):
                if "url" in result["response"]:
                    print(f"Archive URL: {result['response']['url']}")
                if "id" in result["response"]:
                    print(f"Archive ID: {result['response']['id']}")

        return 0

    except LogUploadError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: Failed to upload logs: {e}", file=sys.stderr)
        if args.debug:
            import traceback

            traceback.print_exc()
        return 1


# -----------------------------------------------------------------------------
# CLI Setup
# -----------------------------------------------------------------------------


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="weixin",
        description="Weixin SDK CLI - Manage Weixin bot accounts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  weixin list                              List all accounts
  weixin add bot1 --token abc123           Add a new account
  weixin add bot2 --token xyz789 --name "My Bot"
  weixin show bot1                         Show account details
  weixin update bot1 --name "New Name"     Update account
  weixin enable bot1                       Enable an account
  weixin disable bot1                      Disable an account
  weixin remove bot1                       Remove an account
  weixin test bot1                         Test account configuration
  weixin upload                            Upload logs for troubleshooting
  weixin upload --account bot1             Upload logs for specific account
  weixin upload --date 2026-03-25          Upload logs from specific date

Environment Variables:
  WEIXIN_SDK_STATE_DIR    State directory (default: ~/.weixin_sdk)
  WEIXIN_LOG_UPLOAD_URL   Log upload endpoint URL

For more information, visit: https://github.com/openclaw/openclaw-weixin-python
""",
    )

    # Global options
    parser.add_argument(
        "--version",
        action="version",
        version=f"weixin CLI version {__version__}",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        metavar="DIR",
        help="State directory (default: %(default)s, or $WEIXIN_SDK_STATE_DIR)",
        default=None,
    )
    parser.add_argument(
        "--output",
        choices=[OutputFormat.JSON, OutputFormat.TABLE],
        default=OutputFormat.TABLE,
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )

    # Subcommands
    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Available commands",
    )

    # -------------------------------------------------------------------------
    # list command
    # -------------------------------------------------------------------------
    list_parser = subparsers.add_parser(
        "list",
        help="List all registered accounts",
        description="List all registered accounts",
    )
    list_parser.set_defaults(func=cmd_list)

    # -------------------------------------------------------------------------
    # add command
    # -------------------------------------------------------------------------
    add_parser = subparsers.add_parser(
        "add",
        help="Add a new account",
        description="Add a new account",
    )
    add_parser.add_argument(
        "account_id",
        help="Account ID",
    )
    add_parser.add_argument(
        "--token",
        help="Account token",
    )
    add_parser.add_argument(
        "--name",
        help="Account name",
    )
    add_parser.add_argument(
        "--base-url",
        help="Base API URL (default: https://ilinkai.weixin.qq.com)",
    )
    add_parser.set_defaults(func=cmd_add)

    # -------------------------------------------------------------------------
    # remove command
    # -------------------------------------------------------------------------
    remove_parser = subparsers.add_parser(
        "remove",
        help="Remove an account",
        description="Remove an account",
    )
    remove_parser.add_argument(
        "account_id",
        help="Account ID to remove",
    )
    remove_parser.set_defaults(func=cmd_remove)

    # -------------------------------------------------------------------------
    # enable command
    # -------------------------------------------------------------------------
    enable_parser = subparsers.add_parser(
        "enable",
        help="Enable an account",
        description="Enable an account",
    )
    enable_parser.add_argument(
        "account_id",
        help="Account ID to enable",
    )
    enable_parser.set_defaults(func=cmd_enable)

    # -------------------------------------------------------------------------
    # disable command
    # -------------------------------------------------------------------------
    disable_parser = subparsers.add_parser(
        "disable",
        help="Disable an account",
        description="Disable an account",
    )
    disable_parser.add_argument(
        "account_id",
        help="Account ID to disable",
    )
    disable_parser.set_defaults(func=cmd_disable)

    # -------------------------------------------------------------------------
    # show command
    # -------------------------------------------------------------------------
    show_parser = subparsers.add_parser(
        "show",
        help="Show account details",
        description="Show account details",
    )
    show_parser.add_argument(
        "account_id",
        help="Account ID to show",
    )
    show_parser.set_defaults(func=cmd_show)

    # -------------------------------------------------------------------------
    # test command
    # -------------------------------------------------------------------------
    test_parser = subparsers.add_parser(
        "test",
        help="Test account configuration",
        description="Validate account configuration without making API connections",
    )
    test_parser.add_argument(
        "account_id",
        help="Account ID to test",
    )
    test_parser.set_defaults(func=cmd_test)

    # -------------------------------------------------------------------------
    # update command
    # -------------------------------------------------------------------------
    update_parser = subparsers.add_parser(
        "update",
        help="Update account configuration",
        description="Update account configuration",
    )
    update_parser.add_argument(
        "account_id",
        help="Account ID to update",
    )
    update_parser.add_argument(
        "--token",
        help="New token",
    )
    update_parser.add_argument(
        "--no-token",
        action="store_true",
        help="Remove token",
    )
    update_parser.add_argument(
        "--name",
        help="New name",
    )
    update_parser.add_argument(
        "--base-url",
        help="New base URL",
    )
    update_parser.set_defaults(func=cmd_update)

    # -------------------------------------------------------------------------
    # upload command
    # -------------------------------------------------------------------------
    upload_parser = subparsers.add_parser(
        "upload",
        help="Upload logs for troubleshooting",
        description="Upload logs for troubleshooting",
        epilog="""
Examples:
  weixin upload                           # Upload recent logs
  weixin upload --all                     # Upload all logs
  weixin upload --account bot1            # Upload logs for specific account
  weixin upload --date 2026-03-25         # Upload logs from specific date
  weixin upload --url https://example.com/upload  # Specify upload URL

The upload URL is resolved in this order:
  1. Command line: --url <url>
  2. Environment: WEIXIN_LOG_UPLOAD_URL
  3. Config file: openclaw.json -> logUploadUrl
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    upload_parser.add_argument(
        "--account",
        metavar="ID",
        help="Upload logs for specific account only",
    )
    upload_parser.add_argument(
        "--date",
        metavar="YYYY-MM-DD",
        help="Upload logs from specific date",
    )
    upload_parser.add_argument(
        "--all",
        action="store_true",
        help="Upload all logs (ignores --account and --date filters)",
    )
    upload_parser.add_argument(
        "--url",
        metavar="URL",
        help="Upload endpoint URL (overrides environment/config)",
    )
    upload_parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the ZIP archive after upload (for debugging)",
    )
    upload_parser.set_defaults(func=cmd_upload)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main CLI entry point.

    Args:
        argv: Command line arguments (defaults to sys.argv)

    Returns:
        Exit code
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Handle no command
    if not args.command:
        parser.print_help()
        return 1

    # Determine state directory
    state_dir = args.state_dir or get_state_dir()

    # Initialize account manager
    try:
        manager = WeixinAccountManager(state_dir=state_dir)
    except Exception as e:
        print(f"Error: Failed to initialize account manager: {e}", file=sys.stderr)
        if args.debug:
            import traceback

            traceback.print_exc()
        return 1

    # Execute command
    try:
        return args.func(manager, args)
    except WeixinError as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.debug:
            import traceback

            traceback.print_exc()
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.debug:
            import traceback

            traceback.print_exc()
        return 1


# Entry point for console_scripts
def entry_point() -> int:
    """Entry point for setup.py console_scripts."""
    return main()


if __name__ == "__main__":
    sys.exit(main())
