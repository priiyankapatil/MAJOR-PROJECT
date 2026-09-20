import sys
import json
import os
import re
import time
import requests
import pdfplumber
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from tavily import TavilyClient
except Exception:
    TavilyClient = None

try:
    from groq import Groq
except Exception:
    Groq = None

# ─── CONFIG ───
TAVILY_API_KEY        = os.getenv("TAVILY_API_KEY", "your-tavily-key-here")
GROQ_API_KEY          = os.getenv("GROQ_API_KEY", "your-groq-key-here")
UPDATE_LOG_PATH       = "regulatory_update_log.json"
DYNAMIC_KB_PATH       = "regulatory_dynamic_kb.json"
SCRAPE_TIMEOUT        = 10   # seconds
BORDERLINE_LOW        = 0.40
BORDERLINE_HIGH       = 0.70
MAX_KB_AGE_DAYS       = 30   # warn if KB older than this
LLM_MODEL             = "openai/gpt-oss-20b"

# ─── SCRAPE TARGETS ───
SCRAPE_TARGETS = {
    "CIBRC": {
        "name": "CIB&RC India Banned Pesticides",
        "url": "https://cibrc.nic.in/pbn.php",
        "method": "html",
        "headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
    },
    "EU_SANTE": {
        "name": "EU Pesticide Database",
        "url": "https://food.ec.europa.eu/plants/pesticides/eu-pesticides-database_en",
        "method": "html",
        "headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
    }
}


def load_dynamic_kb() -> dict:
    if os.path.exists(DYNAMIC_KB_PATH):
        try:
            with open(DYNAMIC_KB_PATH, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            pass
    return {
        "version": "1.0",
        "last_updated": None,
        "entries": {},
        "update_history": []
    }


def save_dynamic_kb(kb: dict):
    with open(DYNAMIC_KB_PATH, "w", encoding="utf-8") as handle:
        json.dump(kb, handle, indent=2, ensure_ascii=False)
    print(f"💾 Dynamic KB saved ({len(kb.get('entries', {}))} entries)")


def load_update_log() -> list:
    if os.path.exists(UPDATE_LOG_PATH):
        try:
            with open(UPDATE_LOG_PATH, "r", encoding="utf-8") as handle:
                data = json.load(handle)
                if isinstance(data, list):
                    return data
        except Exception:
            pass
    return []


def append_update_log(entry: dict):
    log = load_update_log()
    payload = {
        "timestamp": datetime.now().isoformat(),
        "type": entry.get("type", "UNKNOWN"),
        "pesticide": entry.get("pesticide"),
        "result": entry.get("result", "UNKNOWN"),
        "source": entry.get("source", "unknown"),
        "action_taken": entry.get("action_taken", "")
    }
    log.append(payload)
    with open(UPDATE_LOG_PATH, "w", encoding="utf-8") as handle:
        json.dump(log, handle, indent=2, ensure_ascii=False)


def check_kb_freshness() -> dict:
    kb = load_dynamic_kb()
    last_updated = kb.get("last_updated")
    if last_updated is None:
        return {"fresh": False, "days_old": None, "warning": "KB never updated — first run"}

    try:
        last_dt = datetime.fromisoformat(last_updated)
        days_old = (datetime.now() - last_dt).days
    except Exception:
        return {"fresh": False, "days_old": None, "warning": "KB timestamp invalid — update recommended"}

    if days_old > MAX_KB_AGE_DAYS:
        return {"fresh": False, "days_old": days_old, "warning": f"KB is {days_old} days old — update recommended"}

    return {"fresh": True, "days_old": days_old, "warning": None}


def _scrape_html_page(source: str, url: str, headers: dict) -> dict:
    response = requests.get(url, headers=headers, timeout=SCRAPE_TIMEOUT)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    tables = soup.find_all("table")
    pesticides_found = []
    raw_count = 0
    keywords = ("banned", "prohibited", "cancelled", "withdrawn", "not approved", "revoked")

    for table in tables:
        text = table.get_text(" ", strip=True)
        if any(keyword in text.lower() for keyword in keywords):
            cells = table.find_all(["td", "th"])
            for cell in cells:
                cell_text = cell.get_text(" ", strip=True)
                raw_count += 1
                if cell_text and len(cell_text) > 2:
                    if re.search(r"[A-Za-z]", cell_text):
                        pesticides_found.append(cell_text[:120])

    unique_found = list(dict.fromkeys(pesticides_found))
    result = {
        "success": True,
        "source": source,
        "url": url,
        "pesticides_found": unique_found,
        "raw_count": raw_count,
        "scraped_at": datetime.now().isoformat()
    }
    print(f"   ✅ {source} scrape succeeded — {len(unique_found)} candidate entries")
    return result


def scrape_cibrc() -> dict:
    target = SCRAPE_TARGETS["CIBRC"]
    try:
        result = _scrape_html_page("CIBRC", target["url"], target["headers"])
        return result
    except Exception as exc:
        result = {
            "success": False,
            "source": "CIBRC",
            "error": str(exc),
            "error_type": type(exc).__name__,
            "fallback_triggered": True,
            "message": "CIB&RC scrape failed — static KB remains authoritative"
        }
        print(f"   ⚠️  CIB&RC scrape failed: {result['error_type']}")
        print(f"   ℹ️  {result['message']}")
        return result


def scrape_eu_sante() -> dict:
    target = SCRAPE_TARGETS["EU_SANTE"]
    try:
        result = _scrape_html_page("EU_SANTE", target["url"], target["headers"])
        return result
    except Exception as exc:
        result = {
            "success": False,
            "source": "EU_SANTE",
            "error": str(exc),
            "error_type": type(exc).__name__,
            "fallback_triggered": True,
            "message": "EU SANTE scrape failed — static KB remains authoritative"
        }
        print(f"   ⚠️  EU SANTE scrape failed: {result['error_type']}")
        print(f"   ℹ️  {result['message']}")
        return result


def search_grounded_llm_verify(pesticide_name: str, tavily_client, groq_client) -> dict:
    print(f"\n   🔍 LLM-grounded verification for: {pesticide_name}")
    try:
        search_query = f"CIB&RC India regulatory status {pesticide_name} pesticide 2024 2025 banned approved"
        search_result = tavily_client.search(
            query=search_query,
            search_depth="basic",
            max_results=3,
            include_answer=False
        )

        snippets = []
        source_urls = []
        for r in search_result.get("results", []):
            snippets.append(f"Source: {r.get('url', '')}\nContent: {r.get('content', '')[:400]}")
            source_urls.append(r.get("url", ""))

        if not snippets:
            return {
                "pesticide": pesticide_name,
                "verified": False,
                "reason": "No search results found",
                "confidence": 0.0,
                "source_url": None
            }

        snippets_text = "\n\n---\n\n".join(snippets)

        print(f"   📄 Search returned {len(snippets)} results")
        print(f"   🔗 Sources: {source_urls}")

    except Exception as exc:
        return {
            "pesticide": pesticide_name,
            "verified": False,
            "reason": f"Tavily search failed: {str(exc)}",
            "confidence": 0.0,
            "source_url": None
        }

    try:
        prompt = f"""You are a regulatory compliance assistant for agricultural pesticides.

Based ONLY on the following recent search results, determine the regulatory status of '{pesticide_name}'.
Do NOT use your training data. Only use what is explicitly stated in these search results.

SEARCH RESULTS:
{snippets_text}

Return ONLY a valid JSON object with NO markdown, NO backticks, NO explanation:
{{
    "pesticide": "{pesticide_name}",
    "india_status": "BANNED or RESTRICTED or APPROVED or UNKNOWN",
    "eu_status": "BANNED or APPROVED or RESTRICTED or UNKNOWN", 
    "recent_changes": "one sentence summary of any changes found, or NONE if nothing found",
    "source_url": "the most relevant URL from search results, or null if none",
    "confidence": 0.0,
    "data_found_in_search": true or false
}}

CRITICAL RULES:
- If search results do not explicitly mention {pesticide_name} regulatory status → set confidence to 0.1 and data_found_in_search to false
- If source_url is null or empty → set confidence to maximum 0.3
- Never invent information not present in the search results
- confidence must reflect ONLY what search results explicitly state"""

        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.0
        )

        raw = response.choices[0].message.content.strip()
        raw = re.sub(r'```json|```', '', raw).strip()
        result = json.loads(raw)

        if not result.get("source_url") or result.get("confidence", 0) < 0.6:
            result["verified"] = False
            result["reason"] = "Insufficient evidence in search results — static KB remains authoritative"
        elif not result.get("data_found_in_search", False):
            result["verified"] = False
            result["reason"] = "Search results did not explicitly mention this pesticide"
        else:
            result["verified"] = True
            result["reason"] = "Verified from search results"

        print(f"   ✅ LLM verification complete")
        print(f"   Confidence  : {result.get('confidence', 0):.2f}")
        print(f"   Verified    : {result.get('verified', False)}")
        print(f"   Source URL  : {result.get('source_url', 'None')}")

        return result

    except json.JSONDecodeError as exc:
        return {
            "pesticide": pesticide_name,
            "verified": False,
            "reason": f"LLM returned invalid JSON: {str(exc)}",
            "confidence": 0.0,
            "source_url": None
        }
    except Exception as exc:
        return {
            "pesticide": pesticide_name,
            "verified": False,
            "reason": f"Groq API error: {str(exc)}",
            "confidence": 0.0,
            "source_url": None
        }


def upsert_dynamic_kb(pesticide_name: str, verified_data: dict, kb: dict) -> dict:
    if not verified_data.get("verified"):
        return kb

    existing = kb.get("entries", {}).get(pesticide_name)
    existing_ts = None
    if existing:
        try:
            existing_ts = datetime.fromisoformat(existing.get("verified_at"))
        except Exception:
            existing_ts = None

    if existing_ts:
        try:
            new_ts = datetime.now()
            if new_ts <= existing_ts:
                return kb
        except Exception:
            pass

    kb.setdefault("entries", {})[pesticide_name] = {
        "india_status": verified_data.get("india_status", "UNKNOWN"),
        "eu_status": verified_data.get("eu_status", "UNKNOWN"),
        "recent_changes": verified_data.get("recent_changes", "NONE"),
        "source_url": verified_data.get("source_url"),
        "confidence": verified_data.get("confidence"),
        "verified_at": datetime.now().isoformat(),
        "supersedes_static_kb": True
    }
    print(f"📝 Dynamic KB updated: {pesticide_name} → {verified_data.get('india_status', 'UNKNOWN')}")
    append_update_log({
        "type": "LLM_VERIFY",
        "pesticide": pesticide_name,
        "result": "UPSERTED",
        "source": verified_data.get("source_url", "none"),
        "action_taken": "UPSERTED"
    })
    return kb


def get_effective_status(pesticide_name: str) -> dict:
    from regulatory_kb import PESTICIDE_DB

    kb = load_dynamic_kb()
    entry = kb.get("entries", {}).get(pesticide_name)
    if entry and entry.get("supersedes_static_kb", False):
        return {
            "source": "DYNAMIC_KB",
            "pesticide": pesticide_name,
            **entry
        }

    static_entry = PESTICIDE_DB.get(pesticide_name, {})
    return {
        "source": "STATIC_KB",
        "pesticide": pesticide_name,
        **static_entry
    }


_LIVE_VERIFICATION_STATE = {
    "cibrc_live_verified": False,
    "eu_sante_live_verified": False,
    "last_checked": None
}


def get_regulatory_verification_status() -> dict:
    """
    Returns explicit structured status of regulatory verification sources.
    Live verification requires actual successful execution of source checks,
    not merely optional dependency availability.
    """
    cibrc_ok = bool(_LIVE_VERIFICATION_STATE.get("cibrc_live_verified", False))
    eu_ok = bool(_LIVE_VERIFICATION_STATE.get("eu_sante_live_verified", False))
    tavily_avail = (TavilyClient is not None)
    groq_avail = (Groq is not None)
    can_update = tavily_avail and groq_avail

    if cibrc_ok and eu_ok:
        mode = "LIVE_VERIFIED"
        disclaimer = "Status verified against live CIB&RC and EU SANTE gazettes."
        freshness = "CHECKED"
    elif cibrc_ok or eu_ok:
        mode = "PARTIAL_LIVE_VERIFIED"
        verified_src = "CIB&RC" if cibrc_ok else "EU SANTE"
        unverified_src = "EU SANTE" if cibrc_ok else "CIB&RC"
        disclaimer = f"Status partially verified against live {verified_src}; {unverified_src} unverified (static KB active)."
        freshness = "PARTIALLY_CHECKED"
    else:
        mode = "OFFLINE_STATIC_KB"
        disclaimer = (
            "Status based on static baseline regulatory KB. Live CIB&RC and EU SANTE checks were skipped "
            "or unverified. Not verified against live gazettes."
        )
        freshness = "UNKNOWN (Live check skipped / unverified)"

    return {
        "live_update_available": can_update,
        "cibrc_live_verified": cibrc_ok,
        "eu_sante_live_verified": eu_ok,
        "mode": mode,
        "freshness": freshness,
        "disclaimer": disclaimer,
    }


def run_startup_update(borderline_pesticides: list = None):
    print("\n" + "="*60)
    print("🔄 REGULATORY KB AUTO-UPDATE STARTING...")
    print("="*60)

    if TavilyClient is None or Groq is None:
        missing = []
        if TavilyClient is None:
            missing.append("tavily")
        if Groq is None:
            missing.append("groq")
        print(f"\n⚠️  Optional updater dependencies unavailable: {', '.join(missing)}")
        print("   ℹ️  Skipping live regulatory refresh; static KB remains authoritative")
        append_update_log({
            "type": "FRESHNESS_CHECK",
            "result": "SKIPPED",
            "source": "local",
            "action_taken": f"Missing optional dependencies: {', '.join(missing)}"
        })
        print("\n" + "="*60)
        print("✅ REGULATORY KB UPDATE COMPLETE")
        print("   CIB&RC scrape  : ⚠️ SKIPPED")
        print("   EU SANTE scrape: ⚠️ SKIPPED")
        print(f"   LLM verifications: {len(borderline_pesticides) if borderline_pesticides else 0}")
        print("   KB freshness   : ⚠️ UNKNOWN")
        print("="*60 + "\n")
        return

    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    groq_client = Groq(api_key=GROQ_API_KEY)

    print("\n📅 Checking KB freshness...")
    freshness = check_kb_freshness()
    if freshness["warning"]:
        print(f"   ⚠️  {freshness['warning']}")
    else:
        print(f"   ✅ KB is fresh ({freshness['days_old']} days old)")

    append_update_log({
        "type": "FRESHNESS_CHECK",
        "result": "FRESH" if freshness["fresh"] else "STALE",
        "source": "local",
        "action_taken": freshness["warning"] or "No action needed"
    })

    print("\n🌐 Running web scrapers...")

    print("   Scraping CIB&RC...")
    cibrc_result = scrape_cibrc()
    if cibrc_result["success"]:
        _LIVE_VERIFICATION_STATE["cibrc_live_verified"] = True
        print(f"   ✅ CIB&RC scraped — {cibrc_result['raw_count']} entries found")
        append_update_log({
            "type": "SCRAPE",
            "result": "SUCCESS",
            "source": "CIBRC",
            "action_taken": f"Found {cibrc_result['raw_count']} pesticide entries"
        })
    else:
        _LIVE_VERIFICATION_STATE["cibrc_live_verified"] = False
        print(f"   ⚠️  CIB&RC scrape failed: {cibrc_result['error_type']}")
        append_update_log({
            "type": "SCRAPE",
            "result": "FAILED",
            "source": "CIBRC",
            "action_taken": f"Fallback to static KB — {cibrc_result['error_type']}"
        })

    print("   Scraping EU SANTE...")
    eu_result = scrape_eu_sante()
    if eu_result["success"]:
        _LIVE_VERIFICATION_STATE["eu_sante_live_verified"] = True
        print(f"   ✅ EU SANTE scraped — {eu_result['raw_count']} entries found")
        append_update_log({
            "type": "SCRAPE",
            "result": "SUCCESS",
            "source": "EU_SANTE",
            "action_taken": f"Found {eu_result['raw_count']} entries"
        })
    else:
        _LIVE_VERIFICATION_STATE["eu_sante_live_verified"] = False
        print(f"   ⚠️  EU SANTE scrape failed: {eu_result['error_type']}")
        append_update_log({
            "type": "SCRAPE",
            "result": "FAILED",
            "source": "EU_SANTE",
            "action_taken": f"Fallback to static KB — {eu_result['error_type']}"
        })

    if borderline_pesticides:
        print(f"\n🧠 Running LLM-grounded verification for {len(borderline_pesticides)} borderline pesticide(s)...")
        kb = load_dynamic_kb()
        for pesticide in borderline_pesticides:
            print(f"\n   Verifying: {pesticide}")
            verified = search_grounded_llm_verify(pesticide, tavily_client, groq_client)
            append_update_log({
                "type": "LLM_VERIFY",
                "pesticide": pesticide,
                "result": "VERIFIED" if verified["verified"] else "UNVERIFIED",
                "source": verified.get("source_url", "none"),
                "action_taken": verified["reason"]
            })
            if verified["verified"]:
                kb = upsert_dynamic_kb(pesticide, verified, kb)
            else:
                print(f"   ⏭️  Skipping upsert — {verified['reason']}")
            time.sleep(2)

        kb["last_updated"] = datetime.now().isoformat()
        save_dynamic_kb(kb)
    else:
        print("\n⏭️  No borderline pesticides to verify this run")

    print("\n" + "="*60)
    print("✅ REGULATORY KB UPDATE COMPLETE")
    print(f"   CIB&RC scrape  : {'✅ SUCCESS' if cibrc_result['success'] else '⚠️ FAILED (static KB active)'}")
    print(f"   EU SANTE scrape: {'✅ SUCCESS' if eu_result['success'] else '⚠️ FAILED (static KB active)'}")
    print(f"   LLM verifications: {len(borderline_pesticides) if borderline_pesticides else 0}")
    print(f"   KB freshness   : {'✅ FRESH' if freshness['fresh'] else '⚠️ STALE'}")
    print("="*60 + "\n")