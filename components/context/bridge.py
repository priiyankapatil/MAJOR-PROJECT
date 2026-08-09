"""Semantic bridge facade for rural-to-scientific normalization."""

from __future__ import annotations


def apply_semantic_bridge(query: str):
    from semantic_bridge import apply_semantic_bridge as _apply
    return _apply(query)
