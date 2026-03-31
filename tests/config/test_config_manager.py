"""
ConfigManager 自测脚本

运行方式:
    python tests/test_config_manager.py
"""

import json
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from middleware.config import ConfigManager


def main() -> None:
    """Self-test: print all key diagnostics and loaded config values."""
    manager = ConfigManager()

    print("[ConfigManager] self-test")
    print(f"  config_package:      {manager._CONFIG_PACKAGE}")
    print(f"  user_config_path:    {manager.user_config_path}")
    print(f"  user_config_dir:     {manager.user_config_dir}")

    schema = manager.load_schema()
    system = manager.load_system_config()
    template = manager.load_user_template()
    print(f"  schema_loaded:       {bool(schema)}")
    print(f"  system_loaded:       {bool(system)}")
    print(f"  user_template_loaded:{bool(template)}")
    print(f"  system_version:      {system.get('version')}")
    print(f"  user_version:        {template.get('version')}")

    manager.ensure_user_config_file()
    print(f"  ensured_config_dir:  {manager.user_config_dir}")
    print(f"  ensured_config_file: {manager.user_config_path}")

    config = manager.load()
    print("  load_status:         OK")

    print("\n[ConfigManager] typed config snapshot")
    print(f"  version:             {config.version}")
    print(f"  app.name:            {config.app.name}")
    print(f"  app.theme:           {config.app.theme}")
    print(f"  app.language:        {config.app.language}")
    print(f"  llm.active_profile:  {config.llm.active_profile}")
    print(f"  llm.current.provider:{config.llm.current.provider}")
    print(f"  llm.current.model:   {config.llm.current.model}")
    print(f"  paths.workspace_dir: {config.paths.workspace_dir}")
    print(f"  db.path:             {manager.get_db_path()}")
    print(f"  db.url:              {manager.get_db_url()}")
    print(f"  paths.skills_dir:    {config.paths.skills_dir}")
    print(f"  skill.path(my-skill):{manager.get_skill_path('my-skill')}")
    print(f"  paths.logs_dir:      {config.paths.logs_dir}")
    print(f"  log.path(app.log):   {manager.get_log_path('app.log')}")
    print(f"  logging.level:       {config.logging.level}")
    print(f"  agent.max_iterations:{config.agent.max_iterations}")

    print("\n[ConfigManager] full JSON")
    print(json.dumps(config.to_json_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
