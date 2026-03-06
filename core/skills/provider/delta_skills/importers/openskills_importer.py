
import shutil
import subprocess
from pathlib import Path

from core.config import g_settings
from core.config.logging import get_logger
from ..schema import SkillImportError

logger = get_logger(__name__)


class OpenSkillsImporter:

    def __init__(self):
        self.skills_dir = g_settings.skills_directory
        self._check_npx_availability()

    @staticmethod
    def _check_npx_availability():
        if not shutil.which("npx"):
            logger.warning("npx is not installed or not in PATH. OpenSkillsImporter will fail.")

    def import_skill(self, source: str, target_dir: Path = None) -> list[Path]:
        logger.info("Importing skill from '%s' using openskills...", source)

        cmd = ["npx", "openskills", "install", source, "--universal", "--yes"]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise SkillImportError(source, f"openskills install failed: {e.stderr}") from e

        agent_skills_dir = Path(".agent/skills")
        if not agent_skills_dir.exists():
            raise SkillImportError(source, "openskills did not create .agent/skills directory.")

        dest_root = target_dir or self.skills_dir
        imported_paths = []
        for item in agent_skills_dir.iterdir():
            if item.is_dir():
                dest_path = dest_root / item.name

                if dest_path.exists():
                    logger.info("Overwriting existing skill at %s", dest_path)
                    shutil.rmtree(dest_path)

                shutil.move(str(item), str(dest_path))
                imported_paths.append(dest_path)
                logger.info("Moved %s to %s", item.name, dest_path)

        if not imported_paths:
            raise SkillImportError(source, "No skills found in .agent/skills after installation.")

        return imported_paths
