"""Phenological gating facade."""

from __future__ import annotations


def apply_phenological_gate(chunks, query, lat, lon):
    from phenology_gate import apply_phenological_gate as _apply
    return _apply(chunks, query, lat, lon)


def print_gate_statistics():
    from phenology_gate import print_gate_statistics as _print
    return _print()
