#!/usr/bin/env python3
"""
Restore parquet file from ChromaDB.
Fixes: parquet file got overwritten during add_new_data.py
Solution: Export all chunks from ChromaDB back to parquet
"""

import os
import pandas as pd
import chromadb
from config import (
    VECTOR_STORE, COLLECTION_NAME,
    ALL_CHUNKS_FILE, PARQUET_ENGINE
)

print("\n" + "="*60)
print("  🔧 RESTORING PARQUET FROM CHROMADB")
print("="*60)

# Connect to ChromaDB
print("\n⏳ Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=VECTOR_STORE)
collection = client.get_collection(COLLECTION_NAME)
print(f"✅ Connected! Total chunks in DB: {collection.count()}")

# Get ALL chunks from ChromaDB
print("\n📥 Exporting chunks from ChromaDB...")
all_data = collection.get(
    include=["documents", "metadatas"]
)

# Reconstruct chunk records
chunks = []
for chunk_id, text, metadata in zip(
    all_data["ids"],
    all_data["documents"],
    all_data["metadatas"]
):
    chunk = {
        "chunk_id"        : chunk_id,
        "text"            : text,
        "source_file"     : metadata.get("source_file", ""),
        "title"           : metadata.get("title", ""),
        "source_org"      : metadata.get("source_org", ""),
        "region"          : metadata.get("region", ""),
        "domain"          : metadata.get("domain", ""),
        "authority_score" : float(metadata.get("authority_score", 0.5)),
        "trust_weight"    : float(metadata.get("trust_weight", 1.0)),
        "crop_tags"       : [
            tag.strip() for tag in 
            metadata.get("crop_tags", "").split(",")
            if tag.strip()
        ],
        "char_count"      : int(metadata.get("char_count", 0)),
    }
    chunks.append(chunk)

print(f"✅ Exported {len(chunks)} chunks")

# Save to parquet
print(f"\n💾 Saving to: {ALL_CHUNKS_FILE}")
df = pd.DataFrame(chunks)

# Backup old file
if os.path.exists(ALL_CHUNKS_FILE):
    backup = ALL_CHUNKS_FILE + ".backup"
    os.rename(ALL_CHUNKS_FILE, backup)
    print(f"   Backed up old file: {backup}")

df.to_parquet(ALL_CHUNKS_FILE, engine=PARQUET_ENGINE, index=False)
restored_size = os.path.getsize(ALL_CHUNKS_FILE) / 1024 / 1024
print(f"   ✅ Parquet file restored: {restored_size:.2f} MB")

# Verify
df_check = pd.read_parquet(ALL_CHUNKS_FILE, engine=PARQUET_ENGINE)
print(f"\n✅ VERIFICATION:")
print(f"   Chunks in file: {len(df_check):,}")
print(f"   Unique sources: {df_check['source_file'].nunique()}")
print(f"   Sources: {', '.join(df_check['source_file'].unique())}")

print("\n" + "="*60)
print("✅ PARQUET RESTORED!")
print("="*60)
