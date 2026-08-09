"""Sentence provenance facade."""

from __future__ import annotations


def build_provenance_map(answer_text, scored_chunks, client):
    from sentence_provenance import build_provenance_map as _build
    return _build(answer_text=answer_text, scored_chunks=scored_chunks, client=client)


def get_provenance_summary(provenance_map):
    from sentence_provenance import get_provenance_summary as _summary
    return _summary(provenance_map)


def print_provenance_report(provenance_map):
    from sentence_provenance import print_provenance_report as _print
    return _print(provenance_map)
