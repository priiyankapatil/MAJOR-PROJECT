"""Entropy and query routing facade."""

from __future__ import annotations


def measure_query_entropy(query):
    from step6_query_gate import measure_query_entropy as _measure
    return _measure(query)


def classify_query(query):
    from step6_query_gate import classify_query as _classify
    return _classify(query)


def make_routing_decision(entropy, classification, threshold=None):
    from step6_query_gate import make_routing_decision as _route
    return _route(entropy, classification, threshold=threshold)


def fast_path_answer(query, chunks):
    from step6_query_gate import fast_path_answer as _fast
    return _fast(query, chunks)


def slow_path_answer(query, chunks, query_type):
    from step6_query_gate import slow_path_answer as _slow
    return _slow(query, chunks, query_type)


def query_gate(query, embedder, collection, bm25, corpus):
    from step6_query_gate import query_gate as _gate
    return _gate(query, embedder, collection, bm25, corpus)
