"""Temporal credibility facade."""

from __future__ import annotations


def compute_temporal_score(source_name, query_type, current_year=None):
    from temporal_credibility import compute_temporal_score as _compute
    return _compute(source_name, query_type, current_year=current_year)


def filter_stale_chunks(scored_chunks, min_score=0.55):
    from temporal_credibility import filter_stale_chunks as _filter
    return _filter(scored_chunks, min_score=min_score)
