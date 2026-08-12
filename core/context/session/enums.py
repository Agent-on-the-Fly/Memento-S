# SPDX-License-Identifier: Apache-2.0
"""core.context.session.enums — session-level runtime state enumerations."""

from enum import StrEnum


class SessionStatus(StrEnum):
    """RuntimeState lifecycle status — single source of truth."""

    AWAITING_USER = "awaiting_user"
    PLANNING = "planning"
    EXECUTING = "executing"
    SEALED = "sealed"
