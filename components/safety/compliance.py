"""Regulatory compliance and updater facade."""

from __future__ import annotations


def scan_answer_for_compliance(answer_text: str):
    from compliance_scanner import scan_answer_for_compliance as _scan
    return _scan(answer_text)


def build_compliance_warning_box(scan_result: dict) -> str:
    from compliance_scanner import build_compliance_warning_box as _build
    return _build(scan_result)


def run_startup_update(borderline_pesticides=None):
    from regulatory_updater import run_startup_update as _run
    return _run(borderline_pesticides=borderline_pesticides)


def load_dynamic_kb():
    from regulatory_updater import load_dynamic_kb as _load
    return _load()
