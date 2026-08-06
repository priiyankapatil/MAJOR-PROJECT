# step5_vector_index.py
# =============================================
# Build BM25 Sparse Index + Hybrid Search
# Combines Dense (ChromaDB) + Sparse (BM25)
# =============================================

import os
import re
import math
import pickle
import pandas as pd
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import chromadb

from config import (
    ALL_CHUNKS_FILE,
    PARQUET_ENGINE,
    VECTOR_STORE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    GRAPH_DIR
)

os.makedirs(GRAPH_DIR, exist_ok=True)

# ── File paths ──
BM25_INDEX_FILE  = os.path.join(GRAPH_DIR, "bm25_index.pkl")
BM25_CORPUS_FILE = os.path.join(GRAPH_DIR, "bm25_corpus.pkl")

# ── Agricultural stop words ──
# Common words in agricultural text that don't help search
AGRI_STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at",
    "to", "for", "of", "with", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might",
    "shall", "can", "need", "this", "that", "these", "those",
    "it", "its", "from", "by", "as", "if", "then", "than",
    "so", "also", "both", "each", "more", "most", "other",
    "into", "through", "during", "before", "after", "above",
    "per", "about", "up", "out", "over", "under",
}


# ─────────────────────────────────────────────
# TEXT PREPROCESSING FOR BM25
# ─────────────────────────────────────────────

def preprocess_for_bm25(text):
    """
    Clean and tokenize text for BM25 indexing.
    
    Steps:
    1. Lowercase everything
    2. Keep agricultural measurements (kg/ha, ml/L)
    3. Remove punctuation except / and .
    4. Split into tokens (words)
    5. Remove stop words
    6. Keep tokens >= 2 chars
    
    Example:
    "Apply neem oil 2ml/L for fruit fly control"
    → ["apply", "neem", "oil", "2ml/l", "fruit", "fly", "control"]
    """
    if not text or not isinstance(text, str):
        return []

    # Lowercase
    text = text.lower()

    # Preserve agricultural measurements
    # "2ml/L" → "2ml_per_l", "10kg/ha" → "10kg_per_ha"
    text = re.sub(r'(\w+)/(\w+)', r'\1_per_\2', text)

    # Remove special chars except alphanumeric, space, underscore
    text = re.sub(r'[^a-z0-9\s_]', ' ', text)

    # Tokenize
    tokens = text.split()

    # Remove stop words and short tokens
    tokens = [
        t for t in tokens
        if t not in AGRI_STOP_WORDS and len(t) >= 2
    ]

    return tokens


# ─────────────────────────────────────────────
# BUILD BM25 INDEX
# ─────────────────────────────────────────────

def build_bm25_index():
    """
    Build BM25 sparse index from all 8,671 chunks.
    
    Saves:
    - bm25_index.pkl  : the BM25 model
    - bm25_corpus.pkl : chunk metadata for retrieval
    """
    print("📂 Loading chunks from parquet...")
    df = pd.read_parquet(
        ALL_CHUNKS_FILE,
        engine  = PARQUET_ENGINE,
        columns = [
            "chunk_id", "text", "source_file",
            "trust_weight", "crop_tags"
        ]
    )
    print(f"   Loaded {len(df):,} chunks\n")

    print("🔤 Tokenizing chunks for BM25...")
    tokenized_corpus = []
    corpus_metadata  = []   # parallel list: metadata per chunk

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc="Tokenizing"):
        tokens = preprocess_for_bm25(row["text"])
        tokenized_corpus.append(tokens)
        corpus_metadata.append({
            "chunk_id"    : row["chunk_id"],
            "text"        : row["text"],
            "source_file" : row["source_file"],
            "trust_weight": float(row["trust_weight"]),
            "crop_tags"   : row["crop_tags"],
        })

    print(f"\n✅ Tokenized {len(tokenized_corpus):,} chunks")
    avg_tokens = sum(len(t) for t in tokenized_corpus) / len(tokenized_corpus)
    print(f"   Average tokens per chunk: {avg_tokens:.1f}")

    print("\n🏗️  Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)
    print("✅ BM25 index built!")

    # Save index
    print("\n💾 Saving BM25 index...")
    with open(BM25_INDEX_FILE, "wb") as f:
        pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save corpus metadata (needed to map scores back to chunks)
    with open(BM25_CORPUS_FILE, "wb") as f:
        pickle.dump(corpus_metadata, f,
                    protocol=pickle.HIGHEST_PROTOCOL)

    bm25_size = os.path.getsize(BM25_INDEX_FILE) / 1024 / 1024
    print(f"   bm25_index.pkl  : {bm25_size:.2f} MB")
    print(f"   bm25_corpus.pkl : saved")

    return bm25, corpus_metadata


def load_bm25_index():
    """Load existing BM25 index from disk."""
    if not os.path.exists(BM25_INDEX_FILE):
        print("❌ BM25 index not found. Run build_bm25_index() first.")
        return None, None

    with open(BM25_INDEX_FILE, "rb") as f:
        bm25 = pickle.load(f)
    with open(BM25_CORPUS_FILE, "rb") as f:
        corpus = pickle.load(f)

    print(f"✅ BM25 index loaded ({len(corpus):,} chunks)")
    return bm25, corpus


# ─────────────────────────────────────────────
# SEARCH FUNCTIONS
# ─────────────────────────────────────────────

def dense_search(query, collection, embedder, top_k=10):
    """
    Dense retrieval using ChromaDB.
    FIXED: correctly extracts chunk_id from metadata.
    """
    query_embedding = embedder.encode([query]).tolist()

    results = collection.query(
        query_embeddings = query_embedding,
        n_results        = top_k,
        include          = ["documents", "metadatas",
                            "distances"]
    )

    dense_results = []
    for doc_id, doc, meta, dist in zip(
        results["ids"][0],           # ← ChromaDB returns ids by default
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        dense_results.append({
            "chunk_id"    : doc_id,  # ← use doc_id not meta chunk_id
            "text"        : doc,
            "source_file" : meta.get("source_file", ""),
            "trust_weight": float(meta.get("trust_weight", 1.0)),
            "dense_score" : round(1 - dist, 4),
            "sparse_score": 0.0,
        })

    return dense_results


def sparse_search(query, bm25, corpus, top_k=10):
    """
    Sparse retrieval using BM25.
    Finds chunks with exact keyword matches.
    
    Returns list of dicts with chunk info + score.
    """
    # Tokenize query same way as corpus
    query_tokens = preprocess_for_bm25(query)

    if not query_tokens:
        return []

    # Get BM25 scores for all chunks
    scores = bm25.get_scores(query_tokens)

    # Get top-k indices
    top_indices = sorted(
        range(len(scores)),
        key    = lambda i: scores[i],
        reverse= True
    )[:top_k]

    sparse_results = []
    max_score = max(scores) if max(scores) > 0 else 1.0

    for idx in top_indices:
        if scores[idx] <= 0:
            continue   # Skip zero-score results

        sparse_results.append({
            "chunk_id"    : corpus[idx]["chunk_id"],
            "text"        : corpus[idx]["text"],
            "source_file" : corpus[idx]["source_file"],
            "trust_weight": corpus[idx]["trust_weight"],
            "dense_score" : 0.0,   # filled in by hybrid search
            "sparse_score": round(scores[idx] / max_score, 4),
        })

    return sparse_results


def hybrid_search(
    query,
    bm25,
    corpus,
    collection,
    embedder,
    top_k      = 5,
    dense_weight  = 0.6,
    sparse_weight = 0.4
):
    """
    Hybrid Retrieval: Dense + Sparse combined.
    
    Formula:
    final_score = (dense_weight × dense_score)
                + (sparse_weight × sparse_score)
                × trust_weight
    
    Args:
        dense_weight  : how much to weight semantic similarity
        sparse_weight : how much to weight keyword matching
        
    Returns top_k results ranked by final score.
    """
    # Step 1: Get results from both systems
    dense_results  = dense_search(query, collection, embedder,
                                  top_k=top_k * 2)
    sparse_results = sparse_search(query, bm25, corpus,
                                   top_k=top_k * 2)

    # Step 2: Merge results by chunk_id
    merged = {}

    for r in dense_results:
        cid = r["chunk_id"]
        merged[cid] = {
            "chunk_id"    : cid,
            "text"        : r["text"],
            "source_file" : r["source_file"],
            "trust_weight": r["trust_weight"],
            "dense_score" : r["dense_score"],
            "sparse_score": 0.0,
        }

    for r in sparse_results:
        cid = r["chunk_id"]
        if cid in merged:
            # Chunk found by BOTH — update sparse score
            merged[cid]["sparse_score"] = r["sparse_score"]
        else:
            # Chunk found only by sparse — add it
            merged[cid] = {
                "chunk_id"    : cid,
                "text"        : r["text"],
                "source_file" : r["source_file"],
                "trust_weight": r["trust_weight"],
                "dense_score" : 0.0,
                "sparse_score": r["sparse_score"],
            }

    # Step 3: Calculate final hybrid score
    # Also apply RA-RAG trust weight here
    # Step 3: Calculate final hybrid score
    for cid, item in merged.items():
        raw_score = (
            dense_weight  * item["dense_score"] +
            sparse_weight * item["sparse_score"]
        )
        # Only apply trust weight if score > 0
        trust = item["trust_weight"] if item["trust_weight"] > 0 else 1.0
        item["final_score"] = round(raw_score * trust, 4)

        # Tag where result came from
        item["found_by"] = []
        if item["dense_score"]  > 0: item["found_by"].append("dense")
        if item["sparse_score"] > 0: item["found_by"].append("sparse")
        if len(item["found_by"]) == 2: item["found_by"] = ["both"]

    # Step 4: Sort by final score and return top_k
    ranked = sorted(
        merged.values(),
        key    = lambda x: x["final_score"],
        reverse= True
    )[:top_k]

    return ranked


# ─────────────────────────────────────────────
# RECIPROCAL RANK FUSION (RRF)
# Alternative to weighted combination
# Better when scores are on different scales
# ─────────────────────────────────────────────

def rrf_hybrid_search(
    query,
    bm25,
    corpus,
    collection,
    embedder,
    top_k = 5,
    k     = 60   # RRF constant (standard default)
):
    """
    Hybrid search using Reciprocal Rank Fusion.
    
    Instead of combining scores directly,
    combines RANKS from each system.
    
    Formula: RRF(doc) = 1/(k + rank_dense)
                      + 1/(k + rank_sparse)
    
    More robust than score combination
    when dense and sparse scores differ in scale.
    """
    # Get ranked lists from both systems
    dense_results  = dense_search(query, collection, embedder,
                                  top_k=top_k * 3)
    sparse_results = sparse_search(query, bm25, corpus,
                                   top_k=top_k * 3)

# Build rank maps — use actual document IDs
    dense_ranks  = {r["chunk_id"]: i+1
                    for i, r in enumerate(dense_results)}
    sparse_ranks = {r["chunk_id"]: i+1
                    for i, r in enumerate(sparse_results)}

    # Verify overlap
    overlap = set(dense_ranks.keys()) & set(sparse_ranks.keys())
    print(f"   [RRF] Dense={len(dense_ranks)} "
          f"Sparse={len(sparse_ranks)} "
          f"Overlap={len(overlap)}")

    # Collect all unique chunk IDs
    all_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())

    # Build full metadata lookup
    all_results = {r["chunk_id"]: r
                   for r in dense_results + sparse_results}

    # Calculate RRF score for each chunk
    rrf_scores = {}
    for cid in all_ids:
        rank_d = dense_ranks.get(cid, top_k * 3 + 1)
        rank_s = sparse_ranks.get(cid, top_k * 3 + 1)
        rrf_scores[cid] = (
            1.0 / (k + rank_d) +
            1.0 / (k + rank_s)
        )

    # Sort by RRF score
    ranked_ids = sorted(
        rrf_scores.keys(),
        key    = lambda cid: rrf_scores[cid],
        reverse= True
    )[:top_k]

    # Build final results
    results = []
    for cid in ranked_ids:
        r = all_results[cid]
        results.append({
            "chunk_id"    : cid,
            "text"        : r["text"],
            "source_file" : r["source_file"],
            "trust_weight": r["trust_weight"],
            "dense_score" : r.get("dense_score", 0.0),
            "sparse_score": r.get("sparse_score", 0.0),
            "rrf_score"   : round(rrf_scores[cid], 6),
            "dense_rank"  : dense_ranks.get(cid, "-"),
            "sparse_rank" : sparse_ranks.get(cid, "-"),
        })

    return results


# ─────────────────────────────────────────────
# TEST AND COMPARE ALL METHODS
# ─────────────────────────────────────────────

def test_all_search_methods(bm25, corpus, collection, embedder):
    """
    Test and compare Dense vs Sparse vs Hybrid vs RRF
    on agricultural queries.
    """
    test_queries = [
        "How to control fruit fly in vegetables?",
        "What is the seed rate for tomato?",
        "Nitrogen deficiency symptoms in crops",
        "Organic manure application for coconut",
        "Irrigation schedule for banana",
    ]

    print("\n" + "="*60)
    print("  🔍 SEARCH METHOD COMPARISON")
    print("="*60)

    for query in test_queries:
        print(f"\n📌 Query: '{query}'")
        print("─"*60)

        # Dense only
        dense = dense_search(
            query, collection, embedder, top_k=3
        )
        print(f"\n  🔵 DENSE (semantic):")
        for r in dense[:2]:
            print(f"     [{r['dense_score']:.3f}] "
                  f"{r['source_file'][:30]}")
            print(f"     {r['text'][:80]}...")

        # Sparse only
        sparse = sparse_search(query, bm25, corpus, top_k=3)
        print(f"\n  🟡 SPARSE (keyword/BM25):")
        if sparse:
            for r in sparse[:2]:
                print(f"     [{r['sparse_score']:.3f}] "
                      f"{r['source_file'][:30]}")
                print(f"     {r['text'][:80]}...")
        else:
            print("     No results above threshold")

        # Hybrid weighted
        # Hybrid weighted
        hybrid = hybrid_search(
            query, bm25, corpus, collection, embedder, top_k=3
        )
        print(f"\n  🟢 HYBRID (dense+sparse weighted):")
        for r in hybrid[:2]:
            found = r.get("found_by", ["?"])
            print(f"     [{r['final_score']:.3f}] "
                  f"dense={r['dense_score']:.3f} "
                  f"sparse={r['sparse_score']:.3f} "
                  f"← found by: {found}")
            print(f"     {r['source_file'][:30]}")
            print(f"     {r['text'][:80]}...")

        # RRF
        rrf = rrf_hybrid_search(
            query, bm25, corpus, collection, embedder, top_k=3
        )
        print(f"\n  🔴 RRF (rank fusion):")
        for r in rrf[:2]:
            print(f"     [rrf={r['rrf_score']:.5f}] "
                  f"d_rank={r['dense_rank']} "
                  f"s_rank={r['sparse_rank']}")
            print(f"     {r['source_file'][:30]}")
            print(f"     {r['text'][:80]}...")

        print()


# ─────────────────────────────────────────────
# SAVE SEARCH ENGINE FOR LATER USE
# ─────────────────────────────────────────────

def save_search_engine_config():
    """
    Save search engine config so other steps
    can load it without rebuilding.
    """
    config = {
        "bm25_index_file" : BM25_INDEX_FILE,
        "bm25_corpus_file": BM25_CORPUS_FILE,
        "dense_weight"    : 0.6,
        "sparse_weight"   : 0.4,
        "rrf_k"           : 60,
        "top_k_default"   : 5,
    }
    config_path = os.path.join(GRAPH_DIR, "search_config.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(config, f)
    print(f"💾 Search config saved: {config_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  📐 VECTOR INDEX — BM25 Sparse Index Build")
    print("="*55)

    # Step 1: Build or load BM25 index (with update detection)
    # Check if chunks have been added since last build
    df_chunks = pd.read_parquet(ALL_CHUNKS_FILE, engine=PARQUET_ENGINE)
    current_chunk_count = len(df_chunks)
    
    if os.path.exists(BM25_INDEX_FILE) and os.path.exists(BM25_CORPUS_FILE):
        # Load existing corpus to check size
        with open(BM25_CORPUS_FILE, "rb") as f:
            old_corpus = pickle.load(f)
        
        if len(old_corpus) == current_chunk_count:
            print(f"✅ BM25 index already exists — loading...")
            print(f"   Index chunks: {len(old_corpus):,} | Parquet chunks: {current_chunk_count:,}")
            bm25, corpus = load_bm25_index()
        else:
            print(f"⚠️  Chunk count mismatch detected!")
            print(f"   BM25 index has: {len(old_corpus):,} chunks")
            print(f"   Parquet file has: {current_chunk_count:,} chunks")
            print(f"   🔄 Rebuilding BM25 index to include new chunks...")
            bm25, corpus = build_bm25_index()
    else:
        print("🏗️  Building new BM25 index...")
        bm25, corpus = build_bm25_index()

    # Step 2: Load Dense search components
    print("\n⏳ Loading embedding model...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    print("✅ Embedding model loaded")

    print("\n⏳ Connecting to ChromaDB...")
    client     = chromadb.PersistentClient(path=VECTOR_STORE)
    collection = client.get_collection(COLLECTION_NAME)
    print(f"✅ ChromaDB connected "
          f"({collection.count():,} chunks)")

    # Step 3: Save config for later steps
    save_search_engine_config()

    # Step 4: Test all methods
    test_all_search_methods(bm25, corpus, collection, embedder)

    print("\n" + "="*55)
    print("✅ VECTOR INDEX COMPLETE!")
    print("   Dense  : ChromaDB (already existed)")
    print("   Sparse : BM25 (newly built)")
    print("   Hybrid : Dense + Sparse combined")
    print("   RRF    : Rank fusion (best method)")
    print("="*55)