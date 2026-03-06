
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class AuditIssue:
    severity: Severity
    category: str
    message: str
    fix_hint: str = ""
    auto_fixable: bool = False


@dataclass
class AuditReport:
    skill_name: str
    skill_dir: Optional[Path] = None
    issues: List[AuditIssue] = field(default_factory=list)
    migrated: bool = False

    @property
    def has_errors(self) -> bool:
        return any(i.severity == Severity.ERROR for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == Severity.WARNING for i in self.issues)

    @property
    def is_clean(self) -> bool:
        return len(self.issues) == 0

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.WARNING)

    def summary(self) -> str:
        if self.is_clean:
            return f"✅ '{self.skill_name}': clean"
        parts = []
        if self.error_count:
            parts.append(f"{self.error_count} error(s)")
        if self.warning_count:
            parts.append(f"{self.warning_count} warning(s)")
        info_count = sum(1 for i in self.issues if i.severity == Severity.INFO)
        if info_count:
            parts.append(f"{info_count} info(s)")
        status = "❌" if self.has_errors else "⚠️"
        return f"{status} '{self.skill_name}': {', '.join(parts)}"
