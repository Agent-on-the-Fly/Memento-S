"""skill store — 技能存储模块（持久化与元数据）

架构：
- base: 抽象基类
- file_storage: 本地文件存储
- vector_storage: 向量数据库存储
- db_storage: DB 元数据存储
- skill_store: 协调层（组合三种存储）

使用示例：
    # 从配置构建（推荐）
    # 内部自动创建 DB、Embedding 等依赖
    store = await SkillStore.from_config(config)

    # 显式构建（测试用）
    store = SkillStore(
        file_storage=FileStorage(skills_dir),
        db_storage=DBStorage(skill_service),
        vector_storage=VectorStorage(db_path)
    )

    # 获取 embedding generator
    generator = store.embedding_generator
    vector = await generator.generate_for_skill(skill)
"""

from core.skill.store.base import SkillStorage
from core.skill.store.file_storage import FileStorage
from core.skill.store.vector_storage import VectorStorage
from core.skill.store.db_storage import DBStorage
from core.skill.store.skill_store import SkillStore

__all__ = [
    "SkillStorage",
    "FileStorage",
    "VectorStorage",
    "DBStorage",
    "SkillStore",
]
