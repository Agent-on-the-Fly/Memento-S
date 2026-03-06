
from core.agent.session_manager import _approx_tokens_from_content


def _count_approx_tokens(content) -> int:
    return _approx_tokens_from_content(content)
