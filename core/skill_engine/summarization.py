"""Step-output summarization helpers."""

from __future__ import annotations

from core.config import STEP_SUMMARY_MAX_TOKENS, STEP_SUMMARY_THRESHOLD
from core.config import DEBUG
from core.llm import get_llm_call_budget, openrouter_messages
from core.utils.logging_utils import log_event
from core.utils.path_utils import _truncate_middle

def _count_approx_tokens(text: str) -> int:
    """Approximate token count using simple heuristic (chars / 4)."""
    return len(str(text or "")) // 4


def summarize_step_output(
    question: str,
    step_skill: str,
    step_output: str,
    max_tokens: int = STEP_SUMMARY_MAX_TOKENS,
    threshold: int = STEP_SUMMARY_THRESHOLD,
) -> str:
    """
    Summarize step output to control context length.
    Only summarizes if output exceeds *threshold* tokens.
    """
    approx_tokens = _count_approx_tokens(step_output)
    if approx_tokens <= threshold:
        return step_output

    if DEBUG:
        log_event(
            "summarize_step_output_start",
            approx_tokens=approx_tokens,
            target_tokens=max_tokens,
        )

    remaining_budget = get_llm_call_budget()
    if isinstance(remaining_budget, int) and remaining_budget <= 1:
        # Preserve budget for routing/planning when the turn is already near limit.
        return _truncate_middle(step_output, max_tokens * 4)

    prompt = f"""You are summarizing a tool execution output for a multi-step task.
The summary will be used as context for subsequent steps, so preserve information needed to answer the question.

=== ORIGINAL QUESTION ===
{question}

=== SKILL USED ===
{step_skill}

=== STEP OUTPUT (needs summarization) ===
{step_output}

=== YOUR TASK ===
1. Extract ONLY information relevant to answering the original question
2. KEEP: specific data, numbers, file paths, URLs, code snippets, error messages, key findings
3. REMOVE: verbose logs, repeated content, progress indicators, irrelevant details
4. If the output contains a final answer or result, preserve it exactly
5. Keep the summary under {max_tokens} tokens

Return ONLY the summarized content. No explanation or meta-commentary."""

    try:
        summary = openrouter_messages(
            "You are a precise summarizer. Return only the essential information.",
            [{"role": "user", "content": prompt}],
        )
        summary = summary.strip()
        if summary:
            summary_tokens = _count_approx_tokens(summary)
            if DEBUG:
                log_event(
                    "summarize_step_output_done",
                    approx_tokens=approx_tokens,
                    summary_tokens=summary_tokens,
                )
            return summary
    except Exception as exc:
        if DEBUG:
            log_event("summarize_step_output_error", error=f"{type(exc).__name__}: {exc}")

    # Fallback: simple truncation if summarization fails
    return _truncate_middle(step_output, max_tokens * 4)
