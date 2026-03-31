"""skill embedding — 向量生成模块

提供统一的 embedding 生成能力，供 store 和 retrieval 模块使用。

使用示例：
    from core.skill.embedding import EmbeddingGenerator
    from core.skill.config import SkillConfig

    # 从配置创建（推荐用于生产环境）
    config = SkillConfig.from_global_config()
    generator = EmbeddingGenerator.from_config(config)

    # 单文本生成
    vector = await generator.generate("查询文本")

    # 为 skill 生成
    vector = await generator.generate_for_skill(skill)

    # 批量生成
    vectors = await generator.generate_for_skills(skills_dict)
"""

from core.skill.embedding.generator import EmbeddingGenerator

__all__ = ["EmbeddingGenerator"]
