
import ast
import re
from pathlib import Path
from typing import List

from core.config.logging import get_logger

from .models import AuditIssue, Severity

logger = get_logger(__name__)

_SANDBOX_ESCAPE_PATTERNS = [
    (r"__builtins__", "Accessing __builtins__ — sandbox escape vector"),
    (r"__subclasses__", "Accessing __subclasses__ — sandbox escape chain"),
    (r"__mro__\s*\[", "MRO traversal — sandbox escape chain"),
    (r"__globals__\s*\[", "Accessing __globals__ — scope escape"),
    (r"__code__\s*\.", "Manipulating __code__ — bytecode tampering"),
]


_DATA_HARVEST_SIGNALS_CRITICAL = [
    r"dict\s*\(\s*os\.environ",                     # dict(os.environ)
    r"os\.environ\s*\.\s*copy\s*\(",                # os.environ.copy()
    r"os\.environ\s*\.\s*items\s*\(",               # os.environ.items()
    r"os\.environ\s*\.\s*keys\s*\(",                # os.environ.keys()
    r"os\.environ\s*\.\s*values\s*\(",              # os.environ.values()
    r"json\.dumps\s*\([^)]*os\.environ",            # json.dumps(os.environ)
    r"str\s*\(\s*os\.environ\s*\)",                 # str(os.environ)
    r"\*\*\s*os\.environ",                           # **os.environ 
    r"for\s+\w+\s+in\s+os\.environ",               # for k in os.environ 
    r"open\s*\(\s*['\"](/etc/passwd|/etc/shadow|~/.ssh)",
    r"glob\.glob\s*\(\s*['\"].*(\.\w+)['\"]",
]

_DATA_HARVEST_SIGNALS_SOFT = [
    r"\.read\(\).*\.encode\(",                       # read + encode ()
    r"open\s*\(\s*['\"].*\.env['\"]",               #  .env 
]

_DATA_EXFIL_SIGNALS = [
    r"requests\.(post|put)\s*\(",
    r"urllib\.request\.(urlopen|Request)",
    r"http\.client\.HTTP",
    r"socket\..*\.send",
    r"smtplib\.",
]

_DESTRUCTIVE_PATTERNS = [
    (r"""(?:rm\s+-rf|rmtree|unlink|remove)\s*\(?\s*['"]?\s*(/|/etc|/usr|/var|/home|/root|\$HOME|~|\.\.)""",
     "Destructive operation targeting system paths"),
    (r"""os\.system\s*\(\s*['\"].*rm\s+-rf\s+/""",
     "Shell command: rm -rf / (system destruction)"),
]

_BACKDOOR_PATTERNS = [
    (r"""open\s*\(\s*['"].*(\.(bashrc|zshrc|bash_profile|profile|zprofile))""",
     "Writing to shell RC file — persistence mechanism"),
    (r"""open\s*\(\s*['"].*(/LaunchAgents/|/LaunchDaemons/|cron)""",
     "Writing to system autostart — persistence mechanism"),
    (r"""open\s*\(\s*['"].*(/etc/cron|/var/spool/cron)""",
     "Writing to crontab — persistence mechanism"),
    (r"""open\s*\(\s*['\"].*(\.ssh/authorized_keys)""",
     "Writing SSH authorized_keys — unauthorized access"),
]

_OBFUSCATION_PATTERNS = [
    (r"base64\.b64decode\s*\(.*\)\s*\)\s*$", "base64 decode + immediate execution — obfuscation"),
    (r"exec\s*\(\s*base64\.", "exec(base64.decode(...)) — obfuscated code execution"),
    (r"exec\s*\(\s*codecs\.decode\(", "exec(codecs.decode(...)) — obfuscated code execution"),
    (r"exec\s*\(\s*bytes\.fromhex\(", "exec(bytes.fromhex(...)) — hex-encoded payload"),
    (r"\\x[0-9a-fA-F]{2}.*\\x[0-9a-fA-F]{2}.*\\x[0-9a-fA-F]{2}.*\\x[0-9a-fA-F]{2}.*"
     r"\\x[0-9a-fA-F]{2}.*\\x[0-9a-fA-F]{2}",
     "Long hex escape sequence — possible obfuscated payload"),
    (r"chr\s*\(\s*\d+\s*\)\s*\+\s*chr\s*\(\s*\d+\s*\)\s*\+\s*chr",
     "chr() concatenation chain — string obfuscation"),
]


class SecurityAuditor:

    @staticmethod
    def audit_code(code: str, skill_name: str = "") -> List[AuditIssue]:
        issues: List[AuditIssue] = []

        if not code or not code.strip():
            return issues

        if code.lstrip().startswith("---"):
            return issues

        _scan_regex_patterns(code, issues)
        _scan_data_exfil(code, issues)
        _scan_ast(code, issues)

        return issues

    @staticmethod
    def audit_directory(skill_dir: Path) -> List[AuditIssue]:
        issues: List[AuditIssue] = []

        for py_file in skill_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            try:
                code = py_file.read_text(encoding="utf-8")
                file_issues = SecurityAuditor.audit_code(code, py_file.name)
                rel_path = py_file.relative_to(skill_dir)
                for issue in file_issues:
                    issue.message = f"[{rel_path}] {issue.message}"
                issues.extend(file_issues)
            except (OSError, UnicodeDecodeError) as e:
                logger.debug("Failed to audit '%s': %s", py_file, e)

        _scan_binary_files(skill_dir, issues)

        return issues


def _scan_regex_patterns(code: str, issues: List[AuditIssue]) -> None:
    _pattern_groups = [
        (_SANDBOX_ESCAPE_PATTERNS, Severity.ERROR, "Sandbox escape",
         "This pattern is almost exclusively used in sandbox escape exploits"),
        (_DESTRUCTIVE_PATTERNS, Severity.ERROR, "Destructive operation",
         "Skill should not target system root or critical directories"),
        (_BACKDOOR_PATTERNS, Severity.ERROR, "Backdoor/persistence",
         "Skills should not modify system startup scripts or SSH keys"),
        (_OBFUSCATION_PATTERNS, Severity.WARNING, "Code obfuscation",
         "Legitimate code rarely needs base64/hex encoding for execution"),
    ]
    for patterns, severity, label, hint in _pattern_groups:
        for pattern, desc in patterns:
            match = re.search(pattern, code)
            if match:
                line_no = code[:match.start()].count("\n") + 1
                issues.append(AuditIssue(
                    severity, "security",
                    f"{label} at line {line_no}: {desc}",
                    fix_hint=hint,
                ))


def _scan_data_exfil(code: str, issues: List[AuditIssue]) -> None:
    critical_hits = sum(1 for p in _DATA_HARVEST_SIGNALS_CRITICAL if re.search(p, code))
    soft_hits = sum(1 for p in _DATA_HARVEST_SIGNALS_SOFT if re.search(p, code))
    exfil_hits = sum(1 for p in _DATA_EXFIL_SIGNALS if re.search(p, code))

    if exfil_hits == 0:
        return

    if critical_hits > 0:
        issues.append(AuditIssue(
            Severity.ERROR, "security",
            f"Potential data exfiltration: code bulk-collects sensitive data "
            f"({critical_hits} critical harvest signal(s)) AND sends to network "
            f"({exfil_hits} exfil signal(s))",
            fix_hint="Bulk env-var access (dict/items/copy) or sensitive file reads "
                     "combined with network calls is a strong exfiltration indicator",
        ))
    elif soft_hits > 0:
        issues.append(AuditIssue(
            Severity.WARNING, "security",
            f"Suspicious data handling: code reads and encodes data "
            f"({soft_hits} soft harvest signal(s)) near network calls "
            f"({exfil_hits} exfil signal(s))",
            fix_hint="Review if file read + encode + network send is legitimate",
        ))


def _scan_ast(code: str, issues: List[AuditIssue]) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.attr, str):
            if node.attr in ("__builtins__", "__subclasses__"):
                issues.append(AuditIssue(
                    Severity.ERROR, "security",
                    f"AST: accessing {node.attr} (line {getattr(node, 'lineno', '?')}) "
                    "— sandbox escape pattern",
                ))

        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "exec" and node.args:
                inner = node.args[0]
                if isinstance(inner, ast.Call):
                    inner_func = inner.func
                    if isinstance(inner_func, ast.Name) and inner_func.id == "compile":
                        issues.append(AuditIssue(
                            Severity.WARNING, "security",
                            f"exec(compile(...)) at line {getattr(node, 'lineno', '?')} "
                            "— dynamic code construction + execution",
                            fix_hint="This pattern is common in obfuscated payloads",
                        ))


def _scan_binary_files(skill_dir: Path, issues: List[AuditIssue]) -> None:
    binary_files = []
    for f in skill_dir.rglob("*"):
        if f.is_file():
            suffix = f.suffix.lower()
            if suffix in (".exe", ".dll", ".so", ".dylib", ".bin", ".com"):
                binary_files.append(f.name)
    if binary_files:
        issues.append(AuditIssue(
            Severity.WARNING, "security",
            f"Pre-compiled binary files: {', '.join(binary_files)}",
            fix_hint="Skills should not bundle native executables — "
                     "use Python scripts or declare system dependencies instead",
        ))
