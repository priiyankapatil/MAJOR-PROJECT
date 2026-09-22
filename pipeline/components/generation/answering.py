"""Answer generation facade."""

from __future__ import annotations


def fast_path_answer(query, chunks):
    from step6_query_gate import fast_path_answer as _fast
    return _fast(query, chunks)


def slow_path_answer(query, chunks, query_type):
    from step6_query_gate import slow_path_answer as _slow
    return _slow(query, chunks, query_type)
