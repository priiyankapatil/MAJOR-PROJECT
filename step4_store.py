# step4_store.py
# =============================================
# PHASE 7: Embed chunks and store in ChromaDB
# =============================================

import os
import pickle
import chromadb
import pandas as pd
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from config import (
    CHUNKS_DIR, VECTOR_STORE,
    COLLECTION_NAME, EMBEDDING_MODEL,
    CHUNKS_STORE_FORMAT, ALL_CHUNKS_FILE, PARQUET_ENGINE
)

# ─────────────────────────────────────────────
# GPU ACCELERATION SETUP
# ─────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print(f"🎮 GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️  GPU not available, using CPU (slower)")

# ── Load embedding model with GPU support ──
print(f"⏳ Loading embedding model: {EMBEDDING_MODEL}")
embedder = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
print(f"✅ Embedding model ready on {DEVICE}!\n")

# ── Setup ChromaDB (persistent = saved to disk) ──
client     = chromadb.PersistentClient(path=VECTOR_STORE)
collection = client.get_or_create_collection(
    name     = COLLECTION_NAME,
    metadata = {"hnsw:space": "cosine"}
)


# step4_store.py — updated store_all_chunks()

def load_chunks():
    if CHUNKS_STORE_FORMAT == "parquet":
        # Check if file exists and is not empty
        if not os.path.exists(ALL_CHUNKS_FILE):
            print("⚠️  No chunk file found. Returning empty list.")
            return []
        
        file_size = os.path.getsize(ALL_CHUNKS_FILE)
        if file_size == 0:
            print("⚠️  Chunk file is empty (0 bytes). Returning empty list.")
            return []
        
        try:
            df = pd.read_parquet(
                ALL_CHUNKS_FILE,
                engine=PARQUET_ENGINE,
                columns=[
                    "chunk_id", "text", "source_file", "title",
                    "source_org", "region", "domain",
                    "authority_score", "trust_weight", "crop_tags",
                    "char_count"
                ]
            )
            return df.to_dict(orient="records")
        except Exception as e:
            print(f"⚠️  Error reading parquet file: {e}")
            return []

    if not os.path.exists(ALL_CHUNKS_FILE):
        return []
        
    chunks = []
    try:
        with open(ALL_CHUNKS_FILE, "rb") as f:
            while True:
                try:
                    chunks.extend(pickle.load(f))
                except EOFError:
                    break
    except Exception as e:
        print(f"⚠️  Error reading pickle file: {e}")
        return []
    
    return chunks


def store_all_chunks():
    all_chunks = load_chunks()

    # Find which chunk IDs are already in ChromaDB
    print("🔍 Checking ChromaDB for existing chunks...")
    existing_ids = set(
        collection.get(include=[])["ids"]
    )
    print(f"   Already stored: {len(existing_ids)} chunks")

    # Filter to only NEW chunks
    new_chunks = [
        c for c in all_chunks
        if c["chunk_id"] not in existing_ids
    ]

    if not new_chunks:
        print("\n✅ ChromaDB is already up to date!")
        print(f"   Total chunks in DB: {collection.count()}")
        return

    print(f"   New chunks to store: {len(new_chunks)}\n")

    # Larger batch size = faster GPU processing
    # RTX 3050 with 6GB VRAM can handle batch_size=128 easily
    BATCH = 128 if DEVICE == "cuda" else 32
    stored = 0

    for i in tqdm(range(0, len(new_chunks), BATCH),
                  desc="Embedding & storing"):
        batch      = new_chunks[i : i + BATCH]
        ids        = [c["chunk_id"] for c in batch]
        texts      = [c["text"]     for c in batch]
        
        # Batch encoding on GPU (much faster)
        embeddings = embedder.encode(
            texts, 
            batch_size=BATCH,
            show_progress_bar=False,
            convert_to_numpy=True,
            device=DEVICE
        ).tolist()
        
        metadatas  = [{
            "source_file"    : c["source_file"],
            "title"          : c["title"],
            "source_org"     : c["source_org"],
            "region"         : c["region"],
            "domain"         : c["domain"],
            "authority_score": c["authority_score"],
            "trust_weight"   : c["trust_weight"],
            "crop_tags"      : ", ".join(c["crop_tags"]),
            "char_count"     : c["char_count"],
        } for c in batch]

        collection.add(
            ids        = ids,
            documents  = texts,
            embeddings = embeddings,
            metadatas  = metadatas,
        )
        stored += len(batch)

    print(f"\n✅ Stored {stored} new chunks!")
    print(f"   Total in ChromaDB now: {collection.count()}")
    print(f"   Device used: {DEVICE}")
    if DEVICE == "cuda":
        print(f"   GPU Memory peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


def test_search(query):
    """Quick test to verify everything works."""
    emb     = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings = emb,
        n_results        = 3,
        include          = ["documents", "metadatas", "distances"]
    )

    print(f"\n🔍 Test Query: '{query}'")
    print("─" * 60)
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        sim = round(1 - dist, 3)
        print(f"  Score : {sim}")
        print(f"  Source: {meta['source_file']}")
        print(f"  Trust : {meta['trust_weight']}")
        print(f"  Text  : {doc[:150]}...")
        print()


if __name__ == "__main__":
    store_all_chunks()

    # Test with 2 sample agricultural queries
    test_search("How to control fruit fly in vegetables?")
    test_search("What is the seed rate for tomato?")