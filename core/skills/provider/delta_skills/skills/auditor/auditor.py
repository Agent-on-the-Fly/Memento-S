
from pathlib import Path
from typing import List

from core.config.logging import get_logger

from .compliance import ComplianceAuditor
from .migrator import SkillMigrator
from .models import AuditIssue, AuditReport, Severity
from .security import SecurityAuditor

logger = get_logger(__name__)


class SkillAuditor:

    def __init__(self, auto_migrate: bool = False):
        self.auto_migrate = auto_migrate

    def audit(self, skill_dir: Path) -> AuditReport:
        report = AuditReport(
            skill_name=skill_dir.name,
            skill_dir=skill_dir,
        )

        report.issues.extend(ComplianceAuditor.audit(skill_dir))
        report.issues.extend(SecurityAuditor.audit_directory(skill_dir))

        if SkillMigrator.needs_migration(skill_dir):
            if self.auto_migrate:
                SkillMigrator.migrate(skill_dir)
                report.migrated = True
                report.issues.append(AuditIssue(
                    Severity.INFO, "migration",
                    "Auto-migrated frontmatter to Anthropic format",
                ))
            else:
                report.issues.append(AuditIssue(
                    Severity.WARNING, "migration",
                    "Frontmatter uses legacy format (non-standard top-level keys)",
                    fix_hint="Run `python -m cli.main audit --migrate` to auto-fix",
                    auto_fixable=True,
                ))

        return report

    def audit_all(self, skills_directory: Path) -> List[AuditReport]:
        reports: List[AuditReport] = []

        if not skills_directory.exists():
            return reports

        for skill_dir in sorted(skills_directory.iterdir()):
            if not skill_dir.is_dir():
                continue
            if skill_dir.name.startswith("."):
                continue
            if not (skill_dir / "SKILL.md").exists():
                continue

            report = self.audit(skill_dir)
            reports.append(report)

            if not report.is_clean:
                logger.info("Audit %s", report.summary())

        return reports

    @staticmethod
    def print_reports(reports: List[AuditReport], verbose: bool = False):
        total = len(reports)
        clean = sum(1 for r in reports if r.is_clean)
        with_errors = sum(1 for r in reports if r.has_errors)
        with_warnings = sum(1 for r in reports if r.has_warnings and not r.has_errors)
        migrated = sum(1 for r in reports if r.migrated)

        print(f"\n{'='*60}")
        print(f"Skill Audit Report — {total} skill(s) scanned")
        print(f"{'='*60}")
        print(f"  ✅ Clean:    {clean}")
        print(f"  ❌ Errors:   {with_errors}")
        print(f"  ⚠️  Warnings: {with_warnings}")
        if migrated:
            print(f"  🔄 Migrated: {migrated}")
        print()

        problem_reports = [r for r in reports if not r.is_clean]
        if not problem_reports and not verbose:
            print("All skills passed audit! 🎉\n")
            return

        for report in reports:
            if report.is_clean and not verbose:
                continue

            print(f"  {report.summary()}")
            if verbose or not report.is_clean:
                for issue in report.issues:
                    icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}[issue.severity.value]
                    print(f"    {icon} [{issue.category}] {issue.message}")
                    if issue.fix_hint:
                        print(f"      → {issue.fix_hint}")
                print()
