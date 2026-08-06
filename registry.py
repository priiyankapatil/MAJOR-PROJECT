# registry.py
# =============================================
# Tracks which PDFs have been processed
# so we never re-process the same file twice
# =============================================

import os
import json
from datetime import date
from config import REGISTRY_FILE, CHUNK_COUNTER_FILE


# ─────────────────────────────────────────────
# REGISTRY: tracks processed files
# ─────────────────────────────────────────────

def load_registry():
    """Load the registry. Create empty one if it doesn't exist."""
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r") as f:
            return json.load(f)
    return {"processed_files": {}}


def save_registry(registry):
    """Save registry to disk."""
    os.makedirs(os.path.dirname(REGISTRY_FILE), exist_ok=True)
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)


def is_already_processed(filename):
    """Check if a PDF has already been fully processed."""
    registry = load_registry()
    entry = registry["processed_files"].get(filename, {})
    return entry.get("status") == "done"


def mark_as_processed(filename, chunks_created, chunk_ids):
    """Mark a PDF as successfully processed."""
    registry = load_registry()
    registry["processed_files"][filename] = {
        "status"         : "done",
        "chunks_created" : chunks_created,
        "processed_date" : str(date.today()),
        "chunk_ids"      : chunk_ids,
    }
    save_registry(registry)
    print(f"   📝 Registered: {filename} "
          f"({chunks_created} chunks)")


def mark_as_failed(filename, error_msg):
    """Mark a PDF as failed so it gets retried next run."""
    registry = load_registry()
    registry["processed_files"][filename] = {
        "status"         : "failed",
        "error"          : error_msg,
        "processed_date" : str(date.today()),
    }
    save_registry(registry)
    print(f"   ❌ Failed: {filename} — {error_msg}")


def get_all_processed():
    """Return list of all successfully processed filenames."""
    registry = load_registry()
    return [
        fname for fname, info
        in registry["processed_files"].items()
        if info.get("status") == "done"
    ]


def get_registry_summary():
    """Print a human-readable summary of the registry."""
    registry = load_registry()
    files    = registry["processed_files"]

    print("\n📋 REGISTRY SUMMARY")
    print("=" * 55)

    done   = [f for f, i in files.items() if i.get("status") == "done"]
    failed = [f for f, i in files.items() if i.get("status") == "failed"]

    for fname in done:
        info = files[fname]
        print(f"  ✅ {fname}")
        print(f"     Chunks: {info.get('chunks_created', '?')} | "
              f"Date: {info.get('processed_date', '?')}")

    for fname in failed:
        info = files[fname]
        print(f"  ❌ {fname}")
        print(f"     Error: {info.get('error', '?')}")

    print(f"\n  Total processed : {len(done)}")
    print(f"  Total failed    : {len(failed)}")
    print("=" * 55)


# ─────────────────────────────────────────────
# CHUNK COUNTER: ensures unique chunk IDs
# even when adding new files later
# ─────────────────────────────────────────────

def get_next_chunk_id():
    """
    Returns the next available chunk ID number.
    Increments and saves automatically.
    
    Example: if 500 chunks exist → returns 500
    Next call → returns 501, etc.
    """
    if os.path.exists(CHUNK_COUNTER_FILE):
        with open(CHUNK_COUNTER_FILE, "r") as f:
            data = json.load(f)
        return data.get("next_id", 0)
    return 0


def advance_chunk_counter(count):
    """
    Advance the counter by 'count' after creating new chunks.
    """
    current = get_next_chunk_id()
    new_val = current + count
    os.makedirs(os.path.dirname(CHUNK_COUNTER_FILE), exist_ok=True)
    with open(CHUNK_COUNTER_FILE, "w") as f:
        json.dump({"next_id": new_val}, f)
    return current   # Returns the START id for this batch