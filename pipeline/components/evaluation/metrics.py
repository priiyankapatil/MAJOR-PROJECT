"""Evaluation scaffold for the final research layer.

This component is intentionally left as the last layer because it depends on
the full system being stable first. Use this module to compare the current
pipeline against a baseline RAG model and to collect metrics for the thesis.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class EvaluationResult:
    query: str
    system_answer: str
    baseline_answer: Optional[str] = None
    reference_answer: Optional[str] = None
    exact_match: Optional[bool] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compare_with_baseline(query: str, system_answer: str,
                          baseline_answer: Optional[str] = None,
                          reference_answer: Optional[str] = None) -> EvaluationResult:
    """Placeholder comparison record for future evaluation experiments."""
    exact_match = None
    if reference_answer is not None:
        exact_match = system_answer.strip() == reference_answer.strip()

    return EvaluationResult(
        query=query,
        system_answer=system_answer,
        baseline_answer=baseline_answer,
        reference_answer=reference_answer,
        exact_match=exact_match,
        notes=(
            "Baseline comparison scaffold. Add faithfulness, relevance, "
            "hallucination, and latency metrics here."
        ),
    )


def summarize_evaluations(results):
    """Summarize a list of EvaluationResult or dict records."""
    items = []
    for item in results:
        if isinstance(item, EvaluationResult):
            items.append(item.to_dict())
        else:
            items.append(dict(item))

    return {
        "count": len(items),
        "items": items,
        "notes": "Evaluation layer scaffold only; plug in metrics and baselines last.",
    }
