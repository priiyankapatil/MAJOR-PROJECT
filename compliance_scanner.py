import re

from regulatory_kb import (
    PESTICIDE_DB, WHO_HAZARD, INDIA_STATUS,
    EU_EXPORT_RISK, ORGANIC_STATUS, ORGANIC_ALTERNATIVES
)

try:
    from regulatory_updater import BORDERLINE_LOW, BORDERLINE_HIGH
except Exception:
    BORDERLINE_LOW = 0.40
    BORDERLINE_HIGH = 0.70


def extract_pesticides_from_text(text: str) -> list:
    matches = []
    seen = set()

    for pesticide_key, data in PESTICIDE_DB.items():
        for name in data.get("common_names", []):
            pattern = rf"\b{re.escape(name)}\b"
            if re.search(pattern, text, re.IGNORECASE):
                if pesticide_key not in seen:
                    seen.add(pesticide_key)
                    matches.append(pesticide_key)
                break

    return matches


def _verdict_from_score(compliance_score: float) -> tuple[str, str]:
    if compliance_score >= 0.80:
        return "SAFE ✅", "green"
    if compliance_score >= 0.60:
        return "CAUTION ⚠️", "yellow"
    if compliance_score >= 0.40:
        return "RESTRICTED ⛔", "orange"
    return "BANNED/UNSAFE 🚫", "red"


def score_pesticide(pesticide_key: str) -> dict:
    data = PESTICIDE_DB[pesticide_key]

    technical_base = 1.0
    who_penalty = WHO_HAZARD[data["who_class"]]["score_penalty"]
    india_penalty = INDIA_STATUS[data["india_status"]]["score_penalty"]
    eu_penalty = EU_EXPORT_RISK[data["eu_export_risk"]]["score_penalty"]
    organic_penalty = ORGANIC_STATUS[data["organic_status"]]["score_penalty"]

    compliance_score = technical_base - (
        0.30 * india_penalty +
        0.35 * eu_penalty +
        0.20 * who_penalty +
        0.15 * organic_penalty
    )
    compliance_score = round(max(0.0, min(1.0, compliance_score)), 4)

    verdict, verdict_color = _verdict_from_score(compliance_score)

    return {
        "pesticide": pesticide_key,
        "common_names": data["common_names"],
        "who_class": data["who_class"],
        "who_label": WHO_HAZARD[data["who_class"]]["label"],
        "india_status": data["india_status"],
        "india_label": INDIA_STATUS[data["india_status"]]["label"],
        "eu_export_risk": data["eu_export_risk"],
        "eu_label": EU_EXPORT_RISK[data["eu_export_risk"]]["label"],
        "organic_status": data["organic_status"],
        "organic_label": ORGANIC_STATUS[data["organic_status"]]["label"],
        "compliance_score": compliance_score,
        "verdict": verdict,
        "verdict_color": verdict_color,
        "pre_harvest_interval": data["pre_harvest_interval_days"],
        "codex_mrl": data["codex_mrl_mg_kg"],
        "notes": data["notes"],
        "organic_alternative": ORGANIC_ALTERNATIVES.get(pesticide_key, "No alternative mapped")
    }


def scan_answer_for_compliance(answer_text: str) -> dict:
    pesticides_found = extract_pesticides_from_text(answer_text)

    if not pesticides_found:
        return {
            "pesticides_found": [],
            "scored_pesticides": [],
            "overall_compliance": 1.0,
            "highest_risk": None,
            "highest_risk_pesticide": None,
            "highest_risk_score": None,
            "warning_needed": False,
            "any_banned": False,
            "any_eu_banned": False,
            "borderline_pesticides": [],
        }

    scored_pesticides = [score_pesticide(key) for key in pesticides_found]
    overall_compliance = round(
        sum(item["compliance_score"] for item in scored_pesticides) / len(scored_pesticides),
        4,
    )
    highest_risk_item = min(scored_pesticides, key=lambda item: item["compliance_score"])

    borderline = [
        p["pesticide"] for p in scored_pesticides
        if BORDERLINE_LOW <= p["compliance_score"] <= BORDERLINE_HIGH
    ]

    return {
        "pesticides_found": pesticides_found,
        "scored_pesticides": scored_pesticides,
        "overall_compliance": overall_compliance,
        "highest_risk": highest_risk_item["pesticide"],
        "highest_risk_pesticide": highest_risk_item["pesticide"],
        "highest_risk_score": highest_risk_item["compliance_score"],
        "warning_needed": overall_compliance < 0.80,
        "any_banned": any(item["india_status"] == "BANNED" for item in scored_pesticides),
        "any_eu_banned": any(item["eu_export_risk"] == "BANNED_EU" for item in scored_pesticides),
        "borderline_pesticides": borderline,
    }


def build_compliance_warning_box(scan_result: dict) -> str:
    pesticides_found = scan_result.get("pesticides_found", [])
    if not pesticides_found:
        return ""

    scored_pesticides = scan_result.get("scored_pesticides", [])
    overall_compliance = scan_result.get("overall_compliance", 1.0)
    overall_verdict, _ = _verdict_from_score(overall_compliance)

    india_banned_names = [item["pesticide"].upper() for item in scored_pesticides if item["india_status"] == "BANNED"]
    eu_banned_names = [item["pesticide"].upper() for item in scored_pesticides if item["eu_export_risk"] == "BANNED_EU"]
    organic_compliant = all(
        item["organic_status"] in {"NPOP_APPROVED", "PGS_APPROVED"}
        for item in scored_pesticides
    )

    phi_values = [item["pre_harvest_interval"] for item in scored_pesticides if item["pre_harvest_interval"] is not None]
    max_phi = max(phi_values) if phi_values else None
    max_phi_text = f"{max_phi} days before harvest" if max_phi is not None else "N/A"

    lines = [
        "╔══════════════════════════════════════════════════════════════╗",
        "║         ⚖️  REGULATORY COMPLIANCE REPORT                    ║",
        "╠══════════════════════════════════════════════════════════════╣",
        f"║  Overall Legal-Efficacy Score : {overall_compliance:.4f} — {overall_verdict}",
        f"║  Pesticides Detected          : {len(pesticides_found)}",
        "╠══════════════════════════════════════════════════════════════╣",
    ]

    for item in scored_pesticides:
        codex_mrl = "N/A" if item["codex_mrl"] is None else f"{item['codex_mrl']} mg/kg"
        phi_text = "N/A" if item["pre_harvest_interval"] is None else f"{item['pre_harvest_interval']} days"

        lines.extend([
            "",
            f"  🔬 {item['pesticide'].upper()}",
            f"     India (CIB&RC)   : {item['india_label']}",
            f"     WHO Hazard Class : {item['who_class']} — {item['who_label']}",
            f"     EU Export Risk   : {item['eu_label']}",
            f"     Organic Status   : {item['organic_label']}",
            f"     Codex MRL        : {codex_mrl}",
            f"     Pre-Harvest Int. : {phi_text}",
            f"     Compliance Score : {item['compliance_score']:.4f}  {item['verdict']}",
            f"     ⚠️  Notes        : {item['notes']}",
            f"     🌿 Organic Alt.  : {item['organic_alternative']}",
        ])

    lines.extend([
        "",
        "╠══════════════════════════════════════════════════════════════╣",
        "║  📋 SUMMARY",
        f"║  🇮🇳 India Banned     : {'Yes (' + ', '.join(india_banned_names) + ')' if india_banned_names else 'No'}",
        f"║  🇪🇺 EU Banned        : {'Yes (' + ', '.join(eu_banned_names) + ')' if eu_banned_names else 'No'}",
        f"║  🌿 Organic Compliant : {'Yes' if organic_compliant else 'No'}",
        f"║  ⏰ Max PHI Required  : {max_phi_text}",
        "╚══════════════════════════════════════════════════════════════╝",
    ])

    return "\n".join(lines)