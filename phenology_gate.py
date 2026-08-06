import requests
import json
import os
from datetime import datetime, timedelta
from stage_compatibility import check_chunk_compatibility, get_compatible_stages
from crop_calendar import get_crop_stage, detect_crop_from_query


def get_satellite_stage(lat: float, lon: float, crop: str) -> dict:
    """Query NASA POWER for recent weather and infer a stage heuristically.

    Returns a dict with source, stage, avg_temp, total_rainfall_30d, confidence
    """
    try:
        end = datetime.utcnow()
        start = end - timedelta(days=30)

        url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        params = {
            "parameters": "ALLSKY_SFC_SW_DWN,T2M,PRECTOTCORR",
            "community": "AG",
            "longitude": lon,
            "latitude": lat,
            "start": start.strftime("%Y%m%d"),
            "end": end.strftime("%Y%m%d"),
            "format": "JSON"
        }

        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        params_data = data.get("properties", {}).get("parameter", {})
        t2m = params_data.get("T2M", {})
        rain = params_data.get("PRECTOTCORR", {})

        temps = [v for v in t2m.values() if v is not None]
        rains = [v for v in rain.values() if v is not None]

        avg_temp = sum(temps) / len(temps) if temps else None
        total_rain = sum(rains) if rains else None

        inferred = None
        confidence = 0.0

        if total_rain is not None and avg_temp is not None:
            # Heuristics
            if total_rain > 100 and 25 <= avg_temp <= 35:
                inferred = "active growth / monsoon stage"
                confidence = 0.7
            elif total_rain < 20 and avg_temp and avg_temp > 35:
                inferred = "dry season management"
                confidence = 0.6
            elif avg_temp < 15:
                inferred = "dormancy or winter stage"
                confidence = 0.6
            else:
                inferred = None
                confidence = 0.0

        return {
            "source": "NASA_SATELLITE",
            "stage": inferred,
            "avg_temp": float(avg_temp) if avg_temp is not None else None,
            "total_rainfall_30d": float(total_rain) if total_rain is not None else None,
            "confidence": confidence
        }

    except Exception:
        return {"source": "NASA_SATELLITE", "stage": None, "confidence": 0.0}


def get_phenological_stage(query: str, lat: float, lon: float) -> dict:
    crop = detect_crop_from_query(query)
    satellite = get_satellite_stage(lat, lon, crop)

    final_stage = None
    source = None

    if satellite.get("confidence", 0) > 0:
        final_stage = satellite.get("stage")
        source = "SATELLITE"
    else:
        cal = get_crop_stage(query)
        final_stage = cal.get("stage")
        source = "CALENDAR"

    month = datetime.now().month

    print("🌿 Phenological Stage Detection:")
    print(f"   Crop    : {crop}")
    print(f"   Stage   : {final_stage}")
    print(f"   Source  : {source} (satellite primary / calendar fallback)")

    return {
        "crop": crop,
        "stage": final_stage,
        "source": source,
        "month": month,
        "satellite_data": satellite
    }


def _log_block(entry: dict, path: str = None):
    if path is None:
        path = os.path.join(os.getcwd(), "phenology_gate_log.json")

    logs = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except Exception:
            logs = []

    logs.append(entry)

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)
    except Exception:
        pass


def apply_phenological_gate(chunks: list, query: str, lat: float, lon: float) -> dict:
    result = get_phenological_stage(query, lat, lon)
    crop = result.get("crop")
    stage = result.get("stage")
    source = result.get("source")

    allowed_chunks = []
    blocked_chunks = []

    for c in chunks:
        text = c.get("text", "")
        comp = check_chunk_compatibility(text, stage)
        if comp.get("compatible"):
            allowed_chunks.append(c)
        else:
            blocked = c.copy()
            blocked["phenology_block"] = comp
            blocked_chunks.append(blocked)

            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "crop": crop,
                "stage": stage,
                "blocked_source": c.get("source_file", c.get("source", "unknown")),
                "blocked_reason": comp.get("reason"),
                "matched_blocked_keyword": comp.get("matched_blocked")
            }
            _log_block(log_entry)

    # Print summary
    print("\n🚧 PHENOLOGICAL GATE RESULTS:")
    print(f"   Crop Stage  : {stage} ({source})")
    print(f"   Total chunks: {len(chunks)}")
    print(f"   ✅ Allowed  : {len(allowed_chunks)}")
    print(f"   🚫 Blocked  : {len(blocked_chunks)}")
    if blocked_chunks:
        print("\n   Blocked chunks:")
        for b in blocked_chunks:
            src = b.get("source_file", b.get("source", "unknown"))
            reason = b.get("phenology_block", {}).get("reason")
            print(f"   🚫 {src} → Reason: {reason}")

    return {
        "allowed_chunks": allowed_chunks,
        "blocked_chunks": blocked_chunks,
        "stage": stage,
        "crop": crop,
        "stage_source": source,
        "total_input": len(chunks),
        "allowed_count": len(allowed_chunks),
        "blocked_count": len(blocked_chunks)
    }


def load_gate_log(path: str = None) -> list:
    if path is None:
        path = os.path.join(os.getcwd(), "phenology_gate_log.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def print_gate_statistics():
    logs = load_gate_log()
    n = len(logs)
    most_blocked_stage = None
    most_blocked_keyword = None
    most_blocked_source = None

    if n == 0:
        print("📊 PHENOLOGICAL GATE STATISTICS\n   No records found.")
        return

    # Simple aggregates
    stage_counts = {}
    keyword_counts = {}
    source_counts = {}

    for l in logs:
        s = l.get("stage")
        kb = l.get("matched_blocked_keyword")
        src = l.get("blocked_source")
        if s:
            stage_counts[s] = stage_counts.get(s, 0) + 1
        if kb:
            keyword_counts[kb] = keyword_counts.get(kb, 0) + 1
        if src:
            source_counts[src] = source_counts.get(src, 0) + 1

    if stage_counts:
        most_blocked_stage = max(stage_counts.items(), key=lambda x: x[1])[0]
    if keyword_counts:
        most_blocked_keyword = max(keyword_counts.items(), key=lambda x: x[1])[0]
    if source_counts:
        most_blocked_source = max(source_counts.items(), key=lambda x: x[1])[0]

    print("📊 PHENOLOGICAL GATE STATISTICS")
    print(f"   Total blocks recorded : {n}")
    print(f"   Most blocked stage    : {most_blocked_stage}")
    print(f"   Most blocked keyword  : {most_blocked_keyword}")
    print(f"   Most blocked source   : {most_blocked_source}")
