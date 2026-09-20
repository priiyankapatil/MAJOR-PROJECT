import re

from regulatory_kb import (
    PESTICIDE_DB, WHO_HAZARD, INDIA_STATUS,
    EU_EXPORT_RISK, ORGANIC_STATUS, ORGANIC_ALTERNATIVES
)

try:
    from regulatory_updater import BORDERLINE_LOW, BORDERLINE_HIGH, get_regulatory_verification_status
except Exception:
    BORDERLINE_LOW = 0.40
    BORDERLINE_HIGH = 0.70
    def get_regulatory_verification_status():
        return {
            "mode": "OFFLINE_STATIC_KB",
            "disclaimer": "Static KB active; live checks skipped."
        }


def extract_pesticides_from_text(text: str) -> list:
    matches = []
    seen = set()

    for pesticide_key, data in PESTICIDE_DB.items():
        for name in data.get("common_names", []):
            pattern = rf"\b{re.escape(name)}\b"
            for m in re.finditer(pattern, text, re.IGNORECASE):
                # An actionable prescription verb or rate indicator near the chemical must NEVER be hidden,
                # even if words like "unverified", "historical", or "withheld" appear nearby.
                start = max(0, m.start() - 80)
                end = min(len(text), m.end() + 80)
                ctx = text[start:end].lower()

                has_actionable_rate_or_verb = any(
                    kw in ctx for kw in [
                        "apply", "spray", "dose", "dosage", "rate", "drench", "use",
                        "%", "ml/l", "l/ha", "kg/ha", "gm/l", "g/l", "ppm", "@", "ec", "wp"
                    ]
                )
                is_purely_blocked = any(
                    phrase in ctx for phrase in [
                        "is blocked", "are blocked", "is withheld", "are withheld",
                        "strictly prohibited", "is banned", "are banned"
                    ]
                )

                # If an actionable prescription is present, it MUST be extracted.
                # Words like "unverified", "historical", or "withheld" cannot hide it.
                if is_purely_blocked and not has_actionable_rate_or_verb:
                    continue

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

    reg_status = get_regulatory_verification_status()
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
        "regulatory_status": reg_status,
    }


def build_compliance_warning_box(scan_result: dict) -> str:
    pesticides_found = scan_result.get("pesticides_found", [])
    if not pesticides_found:
        return ""

    scored_pesticides = scan_result.get("scored_pesticides", [])
    overall_compliance = scan_result.get("overall_compliance", 1.0)
    baseline_verdict, _ = _verdict_from_score(overall_compliance)

    india_banned_names = [item["pesticide"].upper() for item in scored_pesticides if item["india_status"] == "BANNED"]
    eu_banned_names = [item["pesticide"].upper() for item in scored_pesticides if item["eu_export_risk"] == "BANNED_EU"]
    organic_compliant = all(
        item["organic_status"] in {"NPOP_APPROVED", "PGS_APPROVED"}
        for item in scored_pesticides
    )

    phi_values = [item["pre_harvest_interval"] for item in scored_pesticides if item["pre_harvest_interval"] is not None]
    max_phi = max(phi_values) if phi_values else None

    reg_status = scan_result.get("regulatory_status") or get_regulatory_verification_status()
    reg_mode = reg_status.get("mode", "OFFLINE_STATIC_KB")
    reg_disclaimer = reg_status.get("disclaimer", "Live regulatory checks skipped; static reference KB active.")
    is_live_verified = (reg_mode == "LIVE_VERIFIED") and reg_status.get("cibrc_live_verified", False)

    # 1. Determine overall verdict according to verification state
    if is_live_verified:
        overall_verdict_str = f"{overall_compliance:.4f} — {baseline_verdict}"
    elif reg_mode == "UNKNOWN":
        overall_verdict_str = f"{overall_compliance:.4f} — UNKNOWN (Regulatory verification unavailable)"
    elif reg_mode == "PARTIAL_LIVE_VERIFIED":
        overall_verdict_str = f"{overall_compliance:.4f} — PARTIAL (Live verification incomplete; Static KB Baseline)"
    else:  # OFFLINE_STATIC_KB
        if india_banned_names:
            overall_verdict_str = f"{overall_compliance:.4f} — BANNED (Known in Static KB) 🚫"
        elif any(item["india_status"] == "RESTRICTED" for item in scored_pesticides):
            overall_verdict_str = f"{overall_compliance:.4f} — RESTRICTED (Static KB) ⛔"
        elif overall_compliance < 0.60:
            overall_verdict_str = f"{overall_compliance:.4f} — CAUTION / UNVERIFIED ⚠️"
        else:
            overall_verdict_str = f"{overall_compliance:.4f} — PROVISIONAL (Static KB Baseline; Unverified Live)"

    lines = [
        "╔══════════════════════════════════════════════════════════════╗",
        "║         ⚖️  REGULATORY COMPLIANCE REPORT                    ║",
        "╠══════════════════════════════════════════════════════════════╣",
        f"║  Overall Legal-Efficacy Score : {overall_verdict_str}",
        f"║  Pesticides Detected          : {len(pesticides_found)}",
        f"║  Regulatory Verification State : {reg_mode}",
        "╠══════════════════════════════════════════════════════════════╣",
    ]

    for item in scored_pesticides:
        codex_mrl = "N/A" if item["codex_mrl"] is None else f"{item['codex_mrl']} mg/kg"

        if is_live_verified:
            india_status_str = item["india_label"]
            item_score_str = f"{item['compliance_score']:.4f}  {item['verdict']}"
            phi_item_str = f"{item['pre_harvest_interval']} days" if item["pre_harvest_interval"] is not None else "N/A"
            organic_item_str = item["organic_label"]
        elif reg_mode == "UNKNOWN":
            india_status_str = "UNKNOWN (Verification unavailable)"
            item_score_str = f"{item['compliance_score']:.4f}  UNKNOWN (Unverified)"
            phi_item_str = f"{item['pre_harvest_interval']} days (Historical estimate; unverified live)" if item["pre_harvest_interval"] is not None else "N/A"
            organic_item_str = f"{item['organic_label']} (Unverified)"
        elif reg_mode == "PARTIAL_LIVE_VERIFIED":
            cib_ok = reg_status.get("cibrc_live_verified", False)
            if cib_ok:
                india_status_str = item["india_label"]
            else:
                india_status_str = "Recorded as Approved in static KB (⚠️ Live CIB&RC unverified)"

            item_score_str = f"{item['compliance_score']:.4f}  PARTIAL (Live verification incomplete)"
            phi_item_str = (
                f"{item['pre_harvest_interval']} days (Static baseline; verify with product label/KVK)"
                if item["pre_harvest_interval"] is not None else "N/A"
            )
            organic_item_str = f"{item['organic_label']} (Live certification unverified)"
        else:  # OFFLINE_STATIC_KB
            if item["india_status"] == "BANNED":
                india_status_str = "BANNED in India (Recorded in static KB)"
            elif item["india_status"] == "RESTRICTED":
                india_status_str = "RESTRICTED in India (Recorded in static KB)"
            else:
                india_status_str = "Recorded as Approved in static KB (⚠️ Live gazette status unverified)"

            if item["compliance_score"] >= 0.80:
                item_score_str = f"{item['compliance_score']:.4f}  PROVISIONAL (Static Baseline; Unverified Live)"
            else:
                item_score_str = f"{item['compliance_score']:.4f}  {item['verdict']} (Static Baseline)"

            phi_item_str = (
                f"{item['pre_harvest_interval']} days (Static baseline; verify with product label/KVK)"
                if item["pre_harvest_interval"] is not None else "N/A"
            )
            if item["organic_status"] in {"NPOP_APPROVED", "PGS_APPROVED"}:
                organic_item_str = f"{item['organic_label']} in static KB (Live certification unverified)"
            else:
                organic_item_str = f"{item['organic_label']} (Synthetic/Chemical)"

        lines.extend([
            "",
            f"  🔬 {item['pesticide'].upper()}",
            f"     India (CIB&RC)   : {india_status_str}",
            f"     WHO Hazard Class : {item['who_class']} — {item['who_label']}",
            f"     EU Export Risk   : {item['eu_label']}",
            f"     Organic Status   : {organic_item_str}",
            f"     Codex MRL        : {codex_mrl}",
            f"     Pre-Harvest Int. : {phi_item_str}",
            f"     Compliance Score : {item_score_str}",
            f"     ⚠️  Notes        : {item['notes']}",
            f"     🌿 Organic Alt.  : {item['organic_alternative']}",
        ])

    # Summary fields
    if is_live_verified:
        summary_india_banned = f"Yes ({', '.join(india_banned_names)})" if india_banned_names else "No"
        summary_eu_banned = f"Yes ({', '.join(eu_banned_names)})" if eu_banned_names else "No"
        summary_organic = "Yes" if organic_compliant else "No"
        summary_phi = f"{max_phi} days before harvest" if max_phi is not None else "N/A"
    elif reg_mode == "UNKNOWN":
        summary_india_banned = "UNKNOWN (Live verification unavailable)"
        summary_eu_banned = "UNKNOWN (Live verification unavailable)"
        summary_organic = "UNKNOWN"
        summary_phi = f"UNKNOWN (Historical reference: {max_phi} days)" if max_phi is not None else "N/A"
    elif reg_mode == "PARTIAL_LIVE_VERIFIED":
        cib_ok = reg_status.get("cibrc_live_verified", False)
        eu_ok = reg_status.get("eu_sante_live_verified", False)
        summary_india_banned = (
            f"Yes ({', '.join(india_banned_names)})" if (cib_ok and india_banned_names) else
            ("No" if cib_ok else "Not Listed as Banned in Static KB (⚠️ Live CIB&RC unverified; verify with KVK)")
        )
        summary_eu_banned = (
            f"Yes ({', '.join(eu_banned_names)})" if (eu_ok and eu_banned_names) else
            ("No" if eu_ok else "Not Listed as Banned in Static KB (⚠️ Live EU SANTE unverified)")
        )
        summary_organic = (
            "Provisional (Static KB indicates NPOP/PGS; unverified live)"
            if organic_compliant else "No (Synthetic/Chemical)"
        )
        summary_phi = (
            f"Historical reference: {max_phi} days (⚠️ Unverified live; strictly check product label & KVK)"
            if max_phi is not None else "N/A"
        )
    else:
        # Default / OFFLINE_STATIC_KB
        summary_india_banned = (
            f"Yes (Recorded in static KB: {', '.join(india_banned_names)})"
            if india_banned_names else
            "Not Listed as Banned in Static KB (⚠️ Unverified against live CIB&RC gazettes; verify with KVK)"
        )
        summary_eu_banned = (
            f"Yes (Recorded in static KB: {', '.join(eu_banned_names)})"
            if eu_banned_names else
            "Not Listed as Banned in Static KB (⚠️ Unverified against current EU SANTE gazettes)"
        )
        summary_organic = (
            "Provisional (Static KB indicates NPOP/PGS; unverified against live certification lists)"
            if organic_compliant else "No (Synthetic/Chemical)"
        )
        summary_phi = (
            f"Historical reference: {max_phi} days (⚠️ Unverified live; strictly check product label & KVK)"
            if max_phi is not None else "N/A"
        )


    lines.extend([
        "",
        "╠══════════════════════════════════════════════════════════════╣",
        "║  📋 SUMMARY",
        f"║  🇮🇳 India Banned     : {summary_india_banned}",
        f"║  🇪🇺 EU Banned        : {summary_eu_banned}",
        f"║  🌿 Organic Compliant : {summary_organic}",
        f"║  ⏰ Max PHI Required  : {summary_phi}",
        f"║  📡 Verification      : {reg_mode}",
        f"║  ⚠️  Notice           : {reg_disclaimer}",
        "╚══════════════════════════════════════════════════════════════╝",
    ])

    return "\n".join(lines)