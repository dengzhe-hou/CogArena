"""Frozen, paradigm-aware label parser for VLM free-form responses."""

from __future__ import annotations

import re
from typing import Any, Iterable


SCORER_ID = "paradigm-label-parser-v1"


def _label_pattern(labels: Iterable[str]) -> str:
    ordered = sorted(set(labels), key=lambda value: (-len(value), value))
    return "(?:" + "|".join(re.escape(value) for value in ordered) + ")"


def _ordered_hits(text: str, labels: Iterable[str]) -> list[str]:
    hits: list[tuple[int, str]] = []
    for label in labels:
        for match in re.finditer(
            rf"(?<![a-z]){re.escape(label)}(?![a-z])", text, flags=re.IGNORECASE
        ):
            hits.append((match.start(), label))
    seen: set[str] = set()
    output: list[str] = []
    for _, label in sorted(hits):
        if label not in seen:
            seen.add(label)
            output.append(label)
    return output


def _anchor_matches(
    patterns: list[str], text: str, labels: list[str]
) -> list[tuple[str, int, int]]:
    label_re = _label_pattern(labels)
    found: list[tuple[str, int, int]] = []
    for pattern in patterns:
        for match in re.finditer(
            pattern.replace("{LABEL}", f"(?P<label>{label_re})"),
            text,
            flags=re.IGNORECASE,
        ):
            label = match.group("label").lower()
            found.append((label, match.start(), match.end()))
    return found


def _answer_payload_label(payload: str, labels: list[str]) -> tuple[str | None, str]:
    """Parse only the immediate answer value, not labels in its explanation."""

    cleaned = payload.strip().replace("**", "").replace("__", "").strip("`* ")
    if re.match(r"^(?:not|no|neither)\b", cleaned, flags=re.IGNORECASE):
        return None, "malformed"
    match = re.match(
        rf"^(?:the\s+)?(?P<label>{_label_pattern(labels)})"
        rf"(?=$|[\s.,;:!?()\[\]{{}}])",
        cleaned,
        flags=re.IGNORECASE,
    )
    if not match:
        return None, "malformed"
    label = match.group("label").lower()
    tail = cleaned[match.end() :]
    if _has_immediate_alternative(tail, labels):
        return None, "ambiguous"
    return label, "ok"


def _has_immediate_alternative(tail: str, labels: list[str]) -> bool:
    return bool(
        re.match(
            rf"^\s*(?:(?:[,/]\s*(?:(?:and|or)\s*)?)|"
            rf"(?:\(\s*(?:(?:and|or)\s*)?)|(?:(?:and|or)\s+))?"
            rf"(?:then\s+)?(?:the\s+)?{_label_pattern(labels)}\b",
            tail,
            flags=re.IGNORECASE,
        )
    )


def _finish(
    *,
    label: str | None,
    status: str,
    matched_labels: list[str],
    anchor: str | None,
    expected: str,
) -> dict[str, Any]:
    return {
        "label": label,
        "status": status,
        "matched_labels": matched_labels,
        "anchor": anchor,
        "correct": bool(label is not None and label == expected),
    }


def parse_response(
    *,
    paradigm: str,
    response: str,
    expected: str,
    allowed_labels: list[str],
    label_space: list[str],
    query_subject: str | None = None,
) -> dict[str, Any]:
    """Recover one task label without substring or exhaustive-list credit.

    The parser accepts an exact/unique label or one explicit answer anchor.
    Multiple labels without a unique anchor are ambiguous and score zero.
    """

    if not isinstance(response, str):
        raise TypeError("VLM response must be text")
    normalized_allowed = [label.strip().lower() for label in allowed_labels]
    normalized_space = [label.strip().lower() for label in label_space]
    expected = expected.strip().lower()
    if expected not in normalized_allowed:
        raise ValueError(f"expected label {expected!r} is outside task choices")
    if not set(normalized_allowed).issubset(set(normalized_space)):
        raise ValueError("task choices are outside the paradigm label space")

    raw_text = response.strip().lower()
    text = " ".join(raw_text.split())
    if not text:
        return _finish(
            label=None,
            status="blank",
            matched_labels=[],
            anchor=None,
            expected=expected,
        )
    matched = _ordered_hits(text, normalized_space)

    # Exact answer after removing only outer punctuation and Markdown.
    exact_text = raw_text.strip(" \t\r\n`*_#>\"'.,;:!?()[]{}")
    exact_match = re.fullmatch(
        rf"(?:the\s+)?(?P<label>{_label_pattern(normalized_space)})",
        exact_text,
        flags=re.IGNORECASE,
    )
    if exact_match:
        label = exact_match.group("label").lower()
        status = "exact_label" if label in normalized_allowed else "invalid_label"
        return _finish(
            label=label,
            status=status,
            matched_labels=matched,
            anchor="exact",
            expected=expected,
        )

    # A line-anchored explicit answer overrides earlier narrative.  The answer
    # field itself must contain exactly one label, and multiple answer fields
    # must agree.
    answer_fields: list[tuple[str | None, str]] = []
    for line in raw_text.splitlines():
        line = line.strip().replace("**", "").replace("__", "").strip("`* ")
        match = re.match(
            r"^\s*(?:(?:my|the)\s+)?(?:final\s+)?answer\s*(?::|is)\s*(.*)$",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            payload = match.group(1)
            answer_fields.append(
                _answer_payload_label(payload, normalized_space)
            )
    if answer_fields:
        answer_labels = {
            label for label, status in answer_fields if status == "ok" and label
        }
        malformed = (
            len(answer_fields) != 1
            or any(status != "ok" for _, status in answer_fields)
        )
        if malformed or len(answer_labels) != 1:
            return _finish(
                label=None,
                status="ambiguous_answer_field",
                matched_labels=matched,
                anchor="answer_field",
                expected=expected,
            )
        label = next(iter(answer_labels))
        status = "anchored_answer" if label in normalized_allowed else "invalid_label"
        return _finish(
            label=label,
            status=status,
            matched_labels=matched,
            anchor="answer_field",
            expected=expected,
        )

    # Inline "final answer is ..." is accepted only when its payload carries a
    # single label. Generic narrative mentions do not receive credit.
    inline_fields: list[tuple[str | None, str]] = []
    for match in re.finditer(
        r"\b(?:my\s+)?final\s+answer\s*(?:is|:)\s*([^.;!?\n]*)",
        raw_text,
        flags=re.IGNORECASE,
    ):
        inline_fields.append(
            _answer_payload_label(match.group(1), normalized_space)
        )
    if inline_fields:
        inline_labels = {
            label for label, status in inline_fields if status == "ok" and label
        }
        if (
            len(inline_fields) != 1
            or any(status != "ok" for _, status in inline_fields)
            or len(inline_labels) != 1
        ):
            return _finish(
                label=None,
                status="ambiguous_answer_field",
                matched_labels=matched,
                anchor="final_answer",
                expected=expected,
            )
        label = next(iter(inline_labels))
        status = "anchored_answer" if label in normalized_allowed else "invalid_label"
        return _finish(
            label=label,
            status=status,
            matched_labels=matched,
            anchor="final_answer",
            expected=expected,
        )

    if paradigm == "stroop":
        paradigm_patterns = [
            r"\bink(?:\s+color)?\s*(?:is|:|appears?\s+to\s+be)\s*(?:the\s+)?{LABEL}\b",
            r"\b(?:printed|written|displayed|rendered)\s+in\s+(?:the\s+color\s+)?{LABEL}\b",
            r"\bin\s+(?:bold\s+)?{LABEL}\s+(?:letters?|text|ink|font)\b",
            r"\b(?:letters?|font)\s+(?:are|is)\s+{LABEL}\b",
        ]
        anchor_name = "ink_clause"
    elif paradigm == "flanker":
        paradigm_patterns = [
            r"\b(?:cent(?:er|re)|central|middle)(?:\s+arrow|\s+symbol)?\s+(?:points?|faces?|is\s+pointing)\s+(?:to\s+the\s+)?{LABEL}\b",
            r"\b(?:cent(?:er|re)|central|middle)\s+(?:direction|arrow)\s*(?:is|:)\s*(?:the\s+)?{LABEL}\b",
        ]
        anchor_name = "center_clause"
    elif paradigm == "false_belief":
        if not query_subject or not query_subject.strip():
            raise ValueError("false-belief scoring requires the queried character")
        subject = re.escape(query_subject.strip().lower())
        paradigm_patterns = [
            rf"\b{subject}\s+(?:will|would)\s+(?:first\s+)?(?:look|search)(?:\s+for\s+[^.;,]{{0,30}})?\s+(?:in|at|inside)\s+(?:the\s+)?{{LABEL}}\b",
            rf"\b{subject}(?:'s)?\s+first\s+(?:look|search)\s*(?:is|:|will\s+be)\s*(?:in|at|inside)?\s*(?:the\s+)?{{LABEL}}\b",
        ]
        anchor_name = "belief_clause"
    else:
        raise ValueError(f"unsupported VLM paradigm: {paradigm}")

    anchor_matches = _anchor_matches(paradigm_patterns, text, normalized_space)
    anchored = list(dict.fromkeys(match[0] for match in anchor_matches))
    has_immediate_alternative = any(
        _has_immediate_alternative(text[end:], normalized_space)
        for _, _, end in anchor_matches
    )
    if len(anchored) > 1 or has_immediate_alternative:
        return _finish(
            label=None,
            status="ambiguous_anchor",
            matched_labels=matched,
            anchor=anchor_name,
            expected=expected,
        )
    if len(anchored) == 1:
        label = anchored[0]
        status = "anchored_clause" if label in normalized_allowed else "invalid_label"
        return _finish(
            label=label,
            status=status,
            matched_labels=matched,
            anchor=anchor_name,
            expected=expected,
        )
    if not matched:
        return _finish(
            label=None,
            status="no_label",
            matched_labels=[],
            anchor=None,
            expected=expected,
        )
    return _finish(
        label=None,
        status="unanchored_labels",
        matched_labels=matched,
        anchor=None,
        expected=expected,
    )
