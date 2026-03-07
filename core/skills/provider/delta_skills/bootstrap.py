
import shutil
from dataclasses import dataclass
from pathlib import Path

from core.config import g_settings
from core.config.logging import get_logger, setup_logging
from .execution import SkillExecutor
from .resolver import SkillResolver
from .retrieval.cloud_catalog import CloudCatalog
from .skills import SkillAuditor, SkillCreator, SkillLibrary, Severity

logger = get_logger(__name__)

BUILTIN_SKILLS_DIR_NAME = "builtin"
SKILLS_SUBDIR_NAME = "skills"


def _ensure_builtin_skills_synced() -> None:
    project_root = Path(g_settings.project_root)
    builtin_skills_root = project_root / BUILTIN_SKILLS_DIR_NAME / SKILLS_SUBDIR_NAME
    workspace_skills_root = g_settings.skills_directory

    if not builtin_skills_root.is_dir():
        logger.debug("No builtin skills dir at %s, skip sync", builtin_skills_root)
        return

    workspace_skills_root.mkdir(parents=True, exist_ok=True)

    builtin_names = set()
    for d in builtin_skills_root.iterdir():
        if d.is_dir() and not d.name.startswith(".") and (d / "SKILL.md").exists():
            builtin_names.add(d.name)

    to_sync = []
    for name in builtin_names:
        src = builtin_skills_root / name
        dst = workspace_skills_root / name
        if not dst.exists():
            to_sync.append((name, "missing"))
        elif not (dst / "SKILL.md").exists():
            to_sync.append((name, "no_skill_md"))
        elif (src / "scripts").is_dir() and not (dst / "scripts").is_dir():
            to_sync.append((name, "missing_scripts"))

    for name, reason in sorted(to_sync, key=lambda x: x[0]):
        src = builtin_skills_root / name
        dst = workspace_skills_root / name
        try:
            if dst.exists():
                shutil.rmtree(dst)
                logger.info("Removed workspace skill (missing SKILL.md), will replace from builtin: %s", name)
            shutil.copytree(src, dst)
            logger.info("Synced builtin skill to workspace: %s -> %s (%s)", name, dst, reason)
        except Exception as e:
            logger.warning("Failed to copy builtin skill %s: %s", name, e)


@dataclass
class AppContext:

    library: "SkillLibrary"
    creator: "SkillCreator"
    resolver: "SkillResolver"
    executor: "SkillExecutor"
    cloud_catalog: "CloudCatalog"


def _audit_and_migrate(library) -> None:


    if not library.local_cache:
        return

    auditor = SkillAuditor(auto_migrate=True)
    skills_dir = library.skills_directory

    migrated = 0
    security_warnings = 0
    security_errors = 0

    for skill_dir in sorted(skills_dir.iterdir()):
        if not skill_dir.is_dir() or skill_dir.name.startswith("."):
            continue
        if not (skill_dir / "SKILL.md").exists():
            continue

        try:
            report = auditor.audit(skill_dir)

            if report.migrated:
                migrated += 1
                logger.info("Auto-migrated: %s", skill_dir.name)

            for issue in report.issues:
                if issue.category == "security":
                    if issue.severity == Severity.ERROR:
                        security_errors += 1
                        logger.warning(
                            "Security ERROR in '%s': %s",
                            skill_dir.name, issue.message,
                        )
                    elif issue.severity == Severity.WARNING:
                        security_warnings += 1
                        logger.info(
                            "Security warning in '%s': %s",
                            skill_dir.name, issue.message,
                        )
        except Exception as e:
            logger.debug("Audit failed for '%s': %s", skill_dir.name, e)

    total = len(library.local_cache)
    if migrated or security_warnings or security_errors:
        logger.info(
            "Audit complete: %d skill(s), %d migrated, %d security warning(s), %d security error(s)",
            total, migrated, security_warnings, security_errors,
        )
    else:
        logger.info("Audit complete: %d skill(s), all clean", total)


def create_app_context(init_logging: bool = True, llm=None) -> AppContext:
    if init_logging:
        setup_logging(level=g_settings.log_level)

    logger.info(
        "Initializing Delta-Skills Lite (strategy=%s, sandbox=%s)",
        g_settings.resolve_strategy,
        g_settings.sandbox_provider,
    )
    _ensure_builtin_skills_synced()
    library = SkillLibrary()
    _audit_and_migrate(library)

    library._embedding.ensure(library.local_cache, library.skills_directory)
    shared_ef = library._embedding.ef

    catalog_path = g_settings._resolve_path(g_settings.skills_catalog_path)
    cloud_catalog = CloudCatalog(catalog_path)
    if shared_ef:
        cloud_catalog.init_embedding_async(shared_ef)
    else:
        logger.warning("Embedding function not available; cloud catalog will use BM25 only")

    creator = SkillCreator(llm=llm)
    resolver = SkillResolver(library=library, creator=creator, llm=llm)
    executor = resolver.executor
    logger.info(
        "Delta-Skills Lite ready (%d local + %d cloud skills, embedding=%s)",
        len(library.local_cache), cloud_catalog.size,
        "async" if shared_ef else "disabled",
    )
    return AppContext(
        library=library,
        creator=creator,
        resolver=resolver,
        executor=executor,
        cloud_catalog=cloud_catalog,
    )
