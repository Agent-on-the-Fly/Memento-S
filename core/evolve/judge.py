"""Answer judging — binary correctness judge and soft 0-10 scorer."""

from __future__ import annotations

import difflib
import re
from typing import Any

from .tracker import DEFAULT_JUDGE_MODEL
from .text_utils import extract_answer_text, json_fragment, normalize_space

# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_PROMPT = r"""Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available."""


# ---------------------------------------------------------------------------
# Binary correctness judge
# ---------------------------------------------------------------------------

async def judge_answer(
    *,
    question: str,
    answer_type: str,
    model_answer: str,
    expected_answer: str,
    judge_model: str,
    retries: int = 3,
) -> dict[str, Any]:
    from core.evolve import get_evolve_llm

    judge_model_name = str(judge_model or "").strip() or DEFAULT_JUDGE_MODEL
    llm = get_evolve_llm(judge_model_name)

    system = (
        "You are an expert benchmark evaluator.\n"
        "Return exactly one JSON object with keys: extracted_final_answer, reasoning, correct, confidence.\n"
        "No markdown and no extra keys."
    )
    user = JUDGE_PROMPT.format(
        question=question,
        response=model_answer,
        correct_answer=expected_answer,
    )
    messages: list[dict[str, str]] = [{"role": "user", "content": user}]

    last_raw = ""
    last_error = ""
    for attempt in range(max(1, int(retries))):
        try:
            resp = await llm.chat(messages, system=system, max_tokens=4096)
            response = resp.content or ""
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries - 1:
                continue
            return {
                "is_correct": False,
                "score": 0.0,
                "rationale": "judge_llm_error",
                "judge_model": judge_model_name,
                "error": last_error,
            }

        raw = str(response or "").strip()
        last_raw = raw
        parsed = json_fragment(raw)
        if not parsed:
            messages.append(
                {
                    "role": "user",
                    "content": "Return only one JSON object with keys: extracted_final_answer, reasoning, correct, confidence.",
                }
            )
            continue

        is_correct: bool | None = None
        correct_raw = parsed.get("correct")
        if isinstance(correct_raw, str):
            text = correct_raw.strip().lower()
            if text in {"yes", "y", "true", "1"}:
                is_correct = True
            elif text in {"no", "n", "false", "0"}:
                is_correct = False
        elif isinstance(correct_raw, bool):
            is_correct = correct_raw
        elif "is_correct" in parsed:
            is_correct = bool(parsed.get("is_correct"))

        if is_correct is None:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous output missed the `correct` field. "
                        "Return JSON with `correct` as 'yes' or 'no'."
                    ),
                }
            )
            continue

        raw_score = parsed.get("score")
        try:
            if raw_score is None or (isinstance(raw_score, str) and not raw_score.strip()):
                score = 1.0 if is_correct else 0.0
            else:
                score = float(raw_score)
        except Exception:
            score = 1.0 if is_correct else 0.0
        extracted_raw = parsed.get("extracted_final_answer")
        extracted_answer = str(extracted_raw).strip() if extracted_raw is not None else "None"
        if not extracted_answer:
            extracted_answer = "None"
        out = {
            "is_correct": bool(is_correct),
            "score": max(0.0, min(1.0, score)),
            "rationale": str(parsed.get("reasoning") or parsed.get("rationale") or ""),
            "extracted_final_answer": extracted_answer,
            "confidence": str(parsed.get("confidence") or ""),
            "judge_model": judge_model_name,
        }
        if attempt:
            out["judge_retry"] = attempt
        return out

    return {
        "is_correct": False,
        "score": 0.0,
        "rationale": "judge_parse_failed",
        "judge_model": judge_model_name,
        "raw": last_raw,
        "error": last_error,
    }


# ---------------------------------------------------------------------------
# Heuristic soft scorer
# ---------------------------------------------------------------------------

def _soft_text_score_heuristic_0_to_10(expected_answer: str, predicted_answer: str) -> float:
    exp = normalize_space(str(expected_answer or "")).lower()
    pred = normalize_space(str(predicted_answer or "")).lower()
    if not pred:
        return 0.0
    if exp == pred:
        return 10.0

    exp_compact = re.sub(r"[^a-z0-9]+", "", exp)
    pred_compact = re.sub(r"[^a-z0-9]+", "", pred)
    if exp_compact and exp_compact == pred_compact:
        return 9.6

    if exp and pred and (exp in pred or pred in exp):
        short_len = min(len(exp), len(pred))
        long_len = max(len(exp), len(pred))
        ratio = short_len / float(max(1, long_len))
        if ratio >= 0.8:
            return 8.6
        if ratio >= 0.55:
            return 7.2
        return 6.0

    exp_tokens = {x for x in re.findall(r"[a-z0-9]+", exp) if x}
    pred_tokens = {x for x in re.findall(r"[a-z0-9]+", pred) if x}
    token_jaccard = 0.0
    token_containment = 0.0
    if exp_tokens and pred_tokens:
        inter = len(exp_tokens & pred_tokens)
        union = len(exp_tokens | pred_tokens)
        token_jaccard = inter / float(max(1, union))
        token_containment = inter / float(max(1, min(len(exp_tokens), len(pred_tokens))))

    char_sim = difflib.SequenceMatcher(a=exp, b=pred).ratio()
    score = max(token_jaccard * 10.0, token_containment * 8.5, char_sim * 8.5)

    exp_nums = re.findall(r"-?\d+(?:\.\d+)?", exp)
    pred_nums = re.findall(r"-?\d+(?:\.\d+)?", pred)
    if exp_nums and pred_nums and set(exp_nums) != set(pred_nums):
        score = min(score, 5.0)

    return max(0.0, min(10.0, float(score)))


# ---------------------------------------------------------------------------
# Soft 0-10 scorer (used for unit test gates)
# ---------------------------------------------------------------------------

async def judge_answer_soft_score_0_to_10(
    *,
    question: str,
    model_answer: str,
    expected_answer: str,
    judge_model: str,
    retries: int = 3,
    pass_threshold: float = 5.0,
) -> dict[str, Any]:
    from core.evolve import get_evolve_llm

    judge_model_name = str(judge_model or "").strip() or DEFAULT_JUDGE_MODEL
    llm = get_evolve_llm(judge_model_name)
    threshold = max(0.0, min(10.0, float(pass_threshold)))

    system = (
        "You are an evaluator for optimize-stage micro unit tests.\n"
        "Return exactly one JSON object with keys: extracted_final_answer, score_0_to_10, rationale, confidence.\n"
        "Scoring rubric (0-10):\n"
        "- 10: exactly correct or fully equivalent.\n"
        "- 8-9: semantically equivalent with only formatting/site-suffix/casing/abbreviation differences.\n"
        "- 6-7: mostly correct, minor incompleteness but core answer is right.\n"
        "- 0-5: materially incorrect.\n"
        "Be tolerant to punctuation/format differences. No markdown, no extra keys."
    )
    user = (
        "Question:\n"
        f"{question}\n\n"
        "Expected answer:\n"
        f"{expected_answer}\n\n"
        "Model response:\n"
        f"{model_answer}\n\n"
        "Evaluate now."
    )
    messages: list[dict[str, str]] = [{"role": "user", "content": user}]

    fallback_score = _soft_text_score_heuristic_0_to_10(
        expected_answer=expected_answer,
        predicted_answer=extract_answer_text(model_answer),
    )

    last_raw = ""
    last_error = ""
    for attempt in range(max(1, int(retries))):
        try:
            resp = await llm.chat(messages, system=system, max_tokens=800)
            response = resp.content or ""
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries - 1:
                continue
            pass_by_score = bool(fallback_score > threshold)
            return {
                "is_correct": pass_by_score,
                "score": max(0.0, min(1.0, fallback_score / 10.0)),
                "score_0_to_10": float(fallback_score),
                "pass_threshold_0_to_10": float(threshold),
                "rationale": "soft_judge_llm_error_fallback_heuristic",
                "judge_model": judge_model_name,
                "judge_mode": "heuristic_fallback",
                "error": last_error,
            }

        raw = str(response or "").strip()
        last_raw = raw
        parsed = json_fragment(raw)
        if not parsed:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Return only one JSON object with keys: "
                        "extracted_final_answer, score_0_to_10, rationale, confidence."
                    ),
                }
            )
            continue

        extracted_raw = parsed.get("extracted_final_answer")
        extracted_answer = str(extracted_raw).strip() if extracted_raw is not None else ""
        if not extracted_answer:
            extracted_answer = extract_answer_text(model_answer) or "None"

        llm_score_raw = parsed.get("score_0_to_10")
        try:
            llm_score = float(llm_score_raw)
        except Exception:
            llm_score = fallback_score
        llm_score = max(0.0, min(10.0, llm_score))

        final_score = max(llm_score, fallback_score)
        pass_by_score = bool(final_score > threshold)
        out = {
            "is_correct": pass_by_score,
            "score": max(0.0, min(1.0, final_score / 10.0)),
            "score_0_to_10": float(final_score),
            "llm_score_0_to_10": float(llm_score),
            "heuristic_score_0_to_10": float(fallback_score),
            "pass_threshold_0_to_10": float(threshold),
            "rationale": str(parsed.get("rationale") or parsed.get("reasoning") or ""),
            "extracted_final_answer": extracted_answer,
            "confidence": str(parsed.get("confidence") or ""),
            "judge_model": judge_model_name,
            "judge_mode": "soft_0_to_10_blended_max",
        }
        if attempt:
            out["judge_retry"] = attempt
        return out

    pass_by_score = bool(fallback_score > threshold)
    return {
        "is_correct": pass_by_score,
        "score": max(0.0, min(1.0, fallback_score / 10.0)),
        "score_0_to_10": float(fallback_score),
        "pass_threshold_0_to_10": float(threshold),
        "rationale": "soft_judge_parse_failed_fallback_heuristic",
        "judge_model": judge_model_name,
        "judge_mode": "heuristic_fallback",
        "raw": last_raw,
        "error": last_error,
    }
