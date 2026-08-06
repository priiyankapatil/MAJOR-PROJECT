# step5_build_graph.py
# =============================================
# PHASE: Knowledge Graph Building (Pythia-RAG)
# Extracts triplets from chunks using GPT-4
# Builds NetworkX knowledge graph
# =============================================

import os
import json
import time
import pickle
import pandas as pd
import networkx as nx
from tqdm import tqdm
from openai import OpenAI
from config import (
    CHUNKS_DIR, GRAPH_DIR,
    GRAPH_FILE, TRIPLETS_FILE,
    OPENAI_API_KEY, TRIPLET_BATCH_SIZE,
    CHUNKS_STORE_FORMAT, ALL_CHUNKS_FILE, PARQUET_ENGINE
)

os.makedirs(GRAPH_DIR, exist_ok=True)

# ── OpenAI client ──
client = OpenAI(api_key=OPENAI_API_KEY)


# ─────────────────────────────────────────────
# TRIPLET EXTRACTION PROMPT
# Specifically designed for agricultural text
# ─────────────────────────────────────────────

EXTRACTION_PROMPT = """You are an agricultural knowledge extraction expert.

Extract factual relationships from the given agricultural text as triplets.
Each triplet = (subject, relationship, object)

Focus on these relationship types:
- crop → seed_rate → value
- crop → spacing → value  
- crop → season → value
- crop → manure → type_and_quantity
- crop → irrigation → schedule
- pest → affects → crop
- pest → controlled_by → method_or_chemical
- disease → affects → crop
- disease → controlled_by → method
- crop → harvested_at → stage_or_time
- fertilizer → applied_at → stage
- crop → yield → value
- symptom → indicates → disease_or_deficiency
- chemical → dosage → quantity

Rules:
1. Extract ONLY facts clearly stated in the text
2. Keep subjects and objects SHORT (2-5 words max)
3. Relationships must be from the list above or similar
4. Return ONLY valid JSON, nothing else
5. If no clear triplets exist, return empty list

Return format:
{
  "triplets": [
    {
      "subject": "tomato",
      "relationship": "seed_rate",
      "object": "400g per hectare",
      "confidence": 0.95
    }
  ]
}

Agricultural text to analyze:
"""


def extract_triplets_from_chunk(chunk_text, source_file):
    """
    Send one chunk to GPT-4 and get triplets back.
    """
    try:
        response = client.chat.completions.create(
            model    = "gpt-4o-mini",  # cheaper, still accurate
            messages = [
                {
                    "role"   : "system",
                    "content": "You extract agricultural knowledge triplets. "
                               "Return only valid JSON."
                },
                {
                    "role"   : "user",
                    "content": EXTRACTION_PROMPT + chunk_text[:1500]
                }
            ],
            temperature    = 0.1,  # Low temp = consistent extraction
            max_tokens     = 1000,
            response_format= {"type": "json_object"}
        )

        raw = response.choices[0].message.content
        data = json.loads(raw)
        triplets = data.get("triplets", [])

        # Add source info to each triplet
        for t in triplets:
            t["source_file"] = source_file
            t["chunk_text"]  = chunk_text[:200]  # first 200 chars for reference

        return triplets

    except json.JSONDecodeError as e:
        print(f"   ⚠️  JSON parse error: {e}")
        return []
    except Exception as e:
        print(f"   ⚠️  API error: {e}")
        time.sleep(2)  # Wait before retry
        return []


def build_knowledge_graph(all_triplets):
    """
    Build a NetworkX directed graph from triplets.
    
    Nodes = subjects and objects (facts/entities)
    Edges = relationships between them
    """
    G = nx.DiGraph()  # Directed graph

    for triplet in all_triplets:
        subject      = str(triplet["subject"]).strip().lower()
        relationship = str(triplet["relationship"]).strip().lower()
        obj          = str(triplet["object"]).strip().lower()
        confidence   = triplet.get("confidence", 0.8)
        source       = triplet.get("source_file", "")

        # Skip low confidence or empty triplets
        if not subject or not obj or confidence < 0.7:
            continue

        # Add nodes
        if not G.has_node(subject):
            G.add_node(subject, node_type="entity", mentions=1)
        else:
            G.nodes[subject]["mentions"] += 1

        if not G.has_node(obj):
            G.add_node(obj, node_type="value", mentions=1)
        else:
            G.nodes[obj]["mentions"] += 1

        # Add edge (relationship)
        G.add_edge(
            subject, obj,
            relationship = relationship,
            confidence   = confidence,
            source       = source
        )

    return G


def get_graph_stats(G):
    """Print statistics about the knowledge graph."""
    print("\n📊 KNOWLEDGE GRAPH STATISTICS")
    print("=" * 45)
    print(f"  Nodes (entities/values) : {G.number_of_nodes():,}")
    print(f"  Edges (relationships)   : {G.number_of_edges():,}")

    # Most connected nodes (most important concepts)
    degree_sorted = sorted(
        G.degree(), key=lambda x: x[1], reverse=True
    )[:10]

    print(f"\n  🔝 Top 10 most connected nodes:")
    for node, degree in degree_sorted:
        print(f"     {node:<35} connections: {degree}")

    # Relationship type distribution
    rel_types = {}
    for _, _, data in G.edges(data=True):
        rel = data.get("relationship", "unknown")
        rel_types[rel] = rel_types.get(rel, 0) + 1

    print(f"\n  📈 Top relationship types:")
    for rel, count in sorted(
        rel_types.items(), key=lambda x: x[1], reverse=True
    )[:10]:
        print(f"     {rel:<35} count: {count}")

    print("=" * 45)


def load_existing_triplets():
    """Load already extracted triplets (for incremental runs)."""
    if os.path.exists(TRIPLETS_FILE):
        with open(TRIPLETS_FILE, "r") as f:
            return json.load(f)
    return []


def save_triplets(triplets):
    """Save all triplets to disk."""
    with open(TRIPLETS_FILE, "w", encoding="utf-8") as f:
        json.dump(triplets, f, indent=2, ensure_ascii=False)


def save_graph(G):
    """Save NetworkX graph to disk."""
    with open(GRAPH_FILE, "wb") as f:
        pickle.dump(G, f)
    print(f"   💾 Graph saved: {GRAPH_FILE}")


def load_graph():
    """Load existing graph from disk."""
    if os.path.exists(GRAPH_FILE):
        with open(GRAPH_FILE, "rb") as f:
            return pickle.load(f)
    return nx.DiGraph()


def run_graph_building(
    max_chunks=None,
    chunks_per_pdf=150
):
    """
    Main function: extract triplets from chunks
    and build knowledge graph.

    Args:
        max_chunks     : total chunk limit (None = all)
        chunks_per_pdf : how many chunks to sample per PDF
                         (keeps API cost manageable)
    """
    # Load chunks from binary store
    if CHUNKS_STORE_FORMAT == "parquet":
        all_chunks = pd.read_parquet(
            ALL_CHUNKS_FILE,
            engine=PARQUET_ENGINE,
            columns=["chunk_id", "text", "source_file"]
        ).to_dict(orient="records")
    else:
        all_chunks = []
        with open(ALL_CHUNKS_FILE, "rb") as f:
            while True:
                try:
                    all_chunks.extend(pickle.load(f))
                except EOFError:
                    break

    print(f"📦 Total chunks available: {len(all_chunks)}")

    # ── Smart sampling: take best chunks per PDF ──
    # Group by source file
    by_source = {}
    for chunk in all_chunks:
        src = chunk["source_file"]
        by_source.setdefault(src, []).append(chunk)

    # Sample chunks_per_pdf from each source
    sampled = []
    for src, chunks in by_source.items():
        # Take evenly spaced chunks to cover whole document
        step = max(1, len(chunks) // chunks_per_pdf)
        selected = chunks[::step][:chunks_per_pdf]
        sampled.extend(selected)
        print(f"   📄 {src}: {len(selected)} chunks selected")

    if max_chunks:
        sampled = sampled[:max_chunks]

    print(f"\n🎯 Processing {len(sampled)} chunks for triplet extraction")
    print(f"   (Estimated API cost: ~${len(sampled) * 0.001:.2f})\n")

    # Load existing triplets (for incremental runs)
    existing_triplets = load_existing_triplets()
    existing_chunk_ids = {
        t.get("chunk_id", "") for t in existing_triplets
    }
    print(f"📋 Existing triplets: {len(existing_triplets)}")

    # Filter out already processed chunks
    new_chunks = [
        c for c in sampled
        if c["chunk_id"] not in existing_chunk_ids
    ]
    print(f"🆕 New chunks to process: {len(new_chunks)}")

    if not new_chunks:
        print("✅ All chunks already processed!")
    else:
        new_triplets = []
        errors       = 0

        for chunk in tqdm(new_chunks, desc="Extracting triplets"):
            triplets = extract_triplets_from_chunk(
                chunk["text"],
                chunk["source_file"]
            )

            # Tag with chunk_id for incremental tracking
            for t in triplets:
                t["chunk_id"] = chunk["chunk_id"]

            new_triplets.extend(triplets)

            # Rate limiting — avoid hitting API limits
            time.sleep(0.3)

        print(f"\n✅ Extracted {len(new_triplets)} new triplets")
        print(f"   Errors: {errors}")

        # Combine and save
        all_triplets = existing_triplets + new_triplets
        save_triplets(all_triplets)
        print(f"💾 Total triplets saved: {len(all_triplets)}")

    # ── Build graph from ALL triplets ──
    all_triplets = load_existing_triplets()
    print(f"\n🕸️  Building knowledge graph from "
          f"{len(all_triplets)} triplets...")

    G = build_knowledge_graph(all_triplets)
    save_graph(G)
    get_graph_stats(G)

    return G


# ─────────────────────────────────────────────
# GRAPH QUERY FUNCTIONS
# (Used later in retrieval Layer 3)
# ─────────────────────────────────────────────

def query_graph(G, entity, hops=2):
    """
    Find all facts connected to an entity
    within 'hops' steps in the graph.
    
    Example: query_graph(G, "tomato", hops=2)
    Returns all facts about tomato and its neighbors.
    """
    entity = entity.lower().strip()

    if entity not in G:
        # Try partial match
        matches = [
            n for n in G.nodes()
            if entity in n.lower()
        ]
        if not matches:
            return []
        entity = matches[0]

    # Get all nodes within 'hops' distance
    subgraph_nodes = set([entity])
    current_level  = set([entity])

    for _ in range(hops):
        next_level = set()
        for node in current_level:
            # Outgoing edges
            next_level.update(G.successors(node))
            # Incoming edges
            next_level.update(G.predecessors(node))
        subgraph_nodes.update(next_level)
        current_level = next_level

    # Extract all triplets in subgraph
    result_triplets = []
    for u, v, data in G.edges(data=True):
        if u in subgraph_nodes or v in subgraph_nodes:
            result_triplets.append({
                "subject"     : u,
                "relationship": data.get("relationship", ""),
                "object"      : v,
                "source"      : data.get("source", ""),
            })

    return result_triplets


def triplets_to_context(triplets):
    """
    Convert triplets to readable text for LLM context.
    
    Example output:
    "tomato → seed_rate → 400g per hectare (from TNAU Agriculture PDF)
     tomato → spacing → 60cm x 60cm (from TNAU Agriculture PDF)"
    """
    lines = []
    for t in triplets:
        line = (f"{t['subject']} → "
                f"{t['relationship']} → "
                f"{t['object']}")
        if t.get("source"):
            line += f"  [Source: {t['source']}]"
        lines.append(line)
    return "\n".join(lines)


def test_graph_queries(G):
    """Test the graph with sample agricultural queries."""
    test_entities = [
        "tomato", "fruit fly", "nitrogen",
        "coconut", "ginger", "vermicompost"
    ]

    print("\n🔍 TESTING GRAPH QUERIES")
    print("=" * 50)

    for entity in test_entities:
        triplets = query_graph(G, entity, hops=1)
        if triplets:
            print(f"\n  🌱 '{entity}' — {len(triplets)} facts:")
            for t in triplets[:3]:  # Show top 3
                print(f"     {t['subject']} → "
                      f"{t['relationship']} → "
                      f"{t['object']}")
        else:
            print(f"\n  ⚠️  '{entity}' — not found in graph")


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  🕸️  KNOWLEDGE GRAPH BUILDING — Pythia-RAG")
    print("="*55)

    # Build the graph
    # chunks_per_pdf=150 means ~900 chunks total from 6 PDFs
    # Estimated cost: ~$0.90 with gpt-4o-mini
    G = run_graph_building(chunks_per_pdf=150)

    # Test it
    test_graph_queries(G)

    print("\n✅ Knowledge Graph complete!")
    print("   Ready for Layer 3 retrieval (MGRAG + PCST)")