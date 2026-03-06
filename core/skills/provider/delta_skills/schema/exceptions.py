

class DeltaSkillsError(Exception):
    pass


class SkillNotFoundError(DeltaSkillsError):
    def __init__(self, query: str):
        self.query = query
        super().__init__(f"No skill found for query: '{query}'")


class SkillImportError(DeltaSkillsError):
    def __init__(self, skill_name: str, reason: str):
        self.skill_name = skill_name
        self.reason = reason
        super().__init__(f"Failed to import skill '{skill_name}': {reason}")


class SkillExecutionError(DeltaSkillsError):
    def __init__(self, skill_name: str, reason: str):
        self.skill_name = skill_name
        self.reason = reason
        super().__init__(f"Skill '{skill_name}' execution failed: {reason}")


class SkillValidationError(DeltaSkillsError):
    def __init__(self, skill_name: str, reason: str):
        self.skill_name = skill_name
        self.reason = reason
        super().__init__(f"Skill '{skill_name}' validation failed: {reason}")


class SkillCreationError(DeltaSkillsError):
    def __init__(self, skill_name: str, reason: str):
        self.skill_name = skill_name
        self.reason = reason
        super().__init__(f"Failed to create skill '{skill_name}': {reason}")


