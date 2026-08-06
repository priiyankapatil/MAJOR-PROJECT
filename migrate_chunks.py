#!/usr/bin/env python3
"""
Migrate existing all_chunks.json to all_chunks.parquet
Run this once after implementing the binary chunk store.
"""

import os
import json
import pandas as pd
from config import ALL_CHUNKS_FILE, PARQUET_ENGINE

def migrate_chunks_to_parquet():
    """Migrate all_chunks.json to all_chunks.parquet"""
    json_path = ALL_CHUNKS_FILE.replace('.parquet', '.json')

    if not os.path.exists(json_path):
        print(f"❌ No existing JSON file found at: {json_path}")
        return False

    if os.path.exists(ALL_CHUNKS_FILE):
        print(f"⚠️  Parquet file already exists: {ALL_CHUNKS_FILE}")
        print("   Skipping migration to avoid overwriting.")
        return False

    print(f"📦 Migrating {json_path} → {ALL_CHUNKS_FILE}")

    # Load JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    print(f"   Loaded {len(chunks)} chunks from JSON")

    # Convert to DataFrame and save as Parquet
    df = pd.DataFrame(chunks)
    df.to_parquet(ALL_CHUNKS_FILE, engine=PARQUET_ENGINE, index=False)

    print(f"✅ Migration complete: {len(chunks)} chunks saved to Parquet")

    # Optional: backup and remove old file
    backup_path = json_path + '.backup'
    os.rename(json_path, backup_path)
    print(f"   📋 JSON file backed up to: {backup_path}")

    return True

if __name__ == "__main__":
    migrate_chunks_to_parquet()