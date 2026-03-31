"""
Schema 元数据解析模块

解析 JSON Schema 中的 x-* 扩展字段，实现配置字段的权限控制
"""

from typing import Any


class SchemaMetadata:
    """Schema 元数据解析器"""

    @staticmethod
    def is_user_managed(schema: dict, path: str) -> bool:
        """检查指定路径的字段是否由用户管理（模板不自动添加默认值）

        Args:
            schema: JSON Schema 字典
            path: 配置路径，如 "llm.profiles" 或 "app.theme"

        Returns:
            True 如果是用户管理字段
        """
        parts = path.split(".")
        current = schema.get("properties", {})

        for part in parts:
            if not isinstance(current, dict):
                return False

            # 获取当前字段的 schema
            field_schema = current.get(part, {})

            # 检查是否有 x-managed-by: user 标记
            if field_schema.get("x-managed-by") == "user":
                return True

            # 进入下一层
            if "properties" in field_schema:
                current = field_schema["properties"]
            elif "additionalProperties" in field_schema:
                # 对于动态键（如 profiles），检查 additionalProperties
                add_props = field_schema["additionalProperties"]
                if isinstance(add_props, dict):
                    current = add_props.get("properties", {})
            else:
                current = {}

        return False

    @staticmethod
    def get_managed_paths(schema: dict, parent_path: str = "") -> list[str]:
        """获取所有标记为 x-managed-by: user 的路径

        Args:
            schema: JSON Schema 字典
            parent_path: 父路径（递归用）

        Returns:
            用户管理字段的路径列表
        """
        paths = []
        properties = schema.get("properties", {})

        for key, field_schema in properties.items():
            current_path = f"{parent_path}.{key}" if parent_path else key

            # 检查当前字段
            if field_schema.get("x-managed-by") == "user":
                paths.append(current_path)

            # 递归检查嵌套字段
            if "properties" in field_schema:
                paths.extend(
                    SchemaMetadata.get_managed_paths(field_schema, current_path)
                )

        return paths

    @staticmethod
    def merge_respecting_metadata(
        template: dict, user: dict, schema: dict, path: str = ""
    ) -> dict:
        """合并配置，尊重 x-managed-by 标记

        规则：
        1. x-managed-by: user 的字段：完全使用用户值，不合并模板默认值
        2. 其他字段：递归合并，模板补充缺失值
        """
        import copy

        # 如果当前路径是用户管理的，直接返回用户值
        if path and SchemaMetadata.is_user_managed(schema, path):
            return copy.deepcopy(user) if user is not None else {}

        # 都不是字典，返回用户值（如果是基本类型）
        if not isinstance(template, dict) or not isinstance(user, dict):
            return copy.deepcopy(user) if user is not None else copy.deepcopy(template)

        # 递归合并
        result = copy.deepcopy(user)

        for key, template_value in template.items():
            child_path = f"{path}.{key}" if path else key

            if key not in user:
                # 用户没有此键，添加模板值
                result[key] = copy.deepcopy(template_value)
            else:
                # 用户有此键，递归合并
                user_value = user[key]

                # 检查是否是用户管理字段
                if SchemaMetadata.is_user_managed(schema, child_path):
                    # 用户管理字段：保留用户值，不递归合并
                    continue

                # 递归合并嵌套字典
                if isinstance(template_value, dict) and isinstance(user_value, dict):
                    result[key] = SchemaMetadata.merge_respecting_metadata(
                        template_value, user_value, schema, child_path
                    )
                # 其他类型：保留用户值

        return result
