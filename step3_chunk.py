# step3_chunk.py
# =============================================
# PHASE 6: HDR Chunking of all 6 PDFs
# =============================================

import os
import json
import pickle
import re
import torch
import spacy
import pandas as pd
from tqdm import tqdm
from transformers import (
    BertTokenizer,
    BertForNextSentencePrediction
)
from config import (
    EXTRACTED_DIR, CHUNKS_DIR,
    SOURCE_METADATA,
    NSP_THRESHOLD, MAX_CHUNK_SIZE, MIN_CHUNK_SIZE,
    CHUNKS_STORE_FORMAT, ALL_CHUNKS_FILE, PARQUET_ENGINE
)

os.makedirs(CHUNKS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# GPU ACCELERATION SETUP
# ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"🎮 GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️  GPU not available, using CPU (slower)")

# ── Load models ONCE (expensive, do it once) ──
print("⏳ Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

print("⏳ Loading BERT NSP model...")
TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
NSP_MODEL  = BertForNextSentencePrediction.from_pretrained(
    "bert-base-uncased"
)
NSP_MODEL = NSP_MODEL.to(DEVICE)  # Move to GPU if available
NSP_MODEL.eval()
print("✅ Models loaded!\n")


# ─────────────────────────────────────────────
# Agriculture-specific section headers
# These ALWAYS start a new chunk
# ─────────────────────────────────────────────
AGRI_SECTION_HEADERS = re.compile(
    r'^('
    r'Pest[s]?|Disease[s]?|Varieties|Variety|'
    r'Manur(e|ing)|Fertiliz(er|ation)|Nutrient|'
    r'Irrigation|Water\s*Management|'
    r'Harvesting|Harvest|Post.?Harvest|'
    r'Nursery|Sowing|Planting|Transplanting|'
    r'After\s*Cultivation|Inter.?Crop|'
    r'Plant\s*Protection|Weed\s*Control|'
    r'Soil\s*Preparation|Land\s*Preparation|'
    r'Seed\s*Treatment|Crop\s*Protection|'
    r'Storage|Processing|Yield|Economics|'
    r'Climate|Season|Package\s*of\s*Practices'
    r')',
    re.IGNORECASE
)

CROP_NAME_HEADER = re.compile(
    r'^[A-Z][A-Z\s\(\)]{6,}$'  # ALL CAPS lines = crop name headers
)

NUMBERED_SECTION = re.compile(
    r'^\d+[\.\)]\s+[A-Z]'       # "1. Apply..." or "2) Remove..."
)


def is_hard_boundary(sentence):
    """
    Returns True if this sentence MUST start a new chunk.
    Used for agriculture-specific section breaks.
    """
    s = sentence.strip()
    return bool(
        AGRI_SECTION_HEADERS.match(s) or
        CROP_NAME_HEADER.match(s) or
        NUMBERED_SECTION.match(s)
    )


def get_nsp_scores_batch(sentence_pairs):
    """
    Batch BERT inference for multiple sentence pairs (much faster on GPU).
    Takes list of (sent1, sent2) tuples.
    Returns list of scores.
    """
    if not sentence_pairs:
        return []
    
    scores = []
    try:
        sent1s = [sp[0] for sp in sentence_pairs]
        sent2s = [sp[1] for sp in sentence_pairs]
        
        encoding = TOKENIZER(
            sent1s, sent2s,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        )
        
        # Move to GPU if available
        encoding = {k: v.to(DEVICE) for k, v in encoding.items()}
        
        with torch.no_grad():
            logits = NSP_MODEL(**encoding).logits
            probs  = torch.softmax(logits, dim=1)
            scores = probs[:, 0].cpu().numpy().tolist()  # P(IsNext) for each pair
    except Exception as e:
        print(f"   Warning: Batch inference failed: {e}")
        scores = [1.0] * len(sentence_pairs)  # Fallback: assume all connected
    
    return scores


def get_nsp_score(sent1, sent2):
    """
    Ask BERT: does sent2 naturally follow sent1?
    Returns float 0.0–1.0 (higher = more connected).
    Single pair version (uses batch function).
    """
    scores = get_nsp_scores_batch([(sent1, sent2)])
    return scores[0] if scores else 1.0


def hdr_chunk(text):
    """
    HDR Chunking with GPU-accelerated batch processing:
    1. Split text into sentences (spaCy)
    2. Batch process sentence pairs through BERT NSP for efficiency
    3. Group sentences between boundaries into chunks
    """
    # Sentence splitting
    doc       = nlp(text[:1_000_000])   # spaCy limit safety
    sentences = [
        s.text.strip() for s in doc.sents
        if len(s.text.strip()) > 15     # skip tiny fragments
    ]

    if not sentences:
        return []

    # ── Build all sentence pairs we need to check ──
    pairs_to_check = []  # (index, sent1, sent2)
    for i in range(1, len(sentences)):
        pairs_to_check.append((i, sentences[i-1], sentences[i]))

    # ── Process ALL pairs in batch through BERT ──
    if pairs_to_check:
        pair_data = [(p[1], p[2]) for p in pairs_to_check]
        nsp_scores = get_nsp_scores_batch(pair_data)
    else:
        nsp_scores = []

    # ── Build chunks using pre-computed NSP scores ──
    chunks          = []
    current_sents   = [sentences[0]]
    current_len     = len(sentences[0])
    nsp_idx         = 0

    for i in range(1, len(sentences)):
        sent     = sentences[i]
        prev     = sentences[i - 1]

        # ── Decision logic ──────────────────
        hard  = is_hard_boundary(sent)
        big   = (current_len + len(sent)) > MAX_CHUNK_SIZE

        if hard or big:
            # Always split here — no NSP needed
            split = True
        else:
            # Use pre-computed BERT score
            score = nsp_scores[nsp_idx] if nsp_idx < len(nsp_scores) else 1.0
            split = score < NSP_THRESHOLD

        nsp_idx += 1

        # ── Act on decision ─────────────────
        if split:
            chunk_text = " ".join(current_sents).strip()
            if len(chunk_text) >= MIN_CHUNK_SIZE:
                chunks.append(chunk_text)
            current_sents = [sent]
            current_len   = len(sent)
        else:
            current_sents.append(sent)
            current_len  += len(sent)

    # Last chunk
    if current_sents:
        chunk_text = " ".join(current_sents).strip()
        if len(chunk_text) >= MIN_CHUNK_SIZE:
            chunks.append(chunk_text)

    return chunks


def extract_crop_tags(text):
    """
    Tag which crops are mentioned in a chunk.
    Useful for filtered retrieval later.
    """
    CROPS = [
        "rice", "wheat", "maize", "cotton", "sugarcane",
        "tomato", "chilli", "brinjal", "onion", "potato",
        "mango", "banana", "coconut", "cashew", "tapioca",
        "bittergourd", "cucumber", "pumpkin", "okra",
        "ginger", "turmeric", "cardamom", "pepper",
        "groundnut", "soybean", "sunflower", "mustard",
        "tea", "coffee", "rubber", "arecanut",
        "jasmine", "marigold", "rose", "orchid",
    ]
    tl = text.lower()
    return [c for c in CROPS if c in tl]


def save_chunks_batch(chunks, chunks_path):
    """Persist a list of chunk records to the configured binary store."""
    if not chunks:
        return

    if CHUNKS_STORE_FORMAT == "parquet":
        df = pd.DataFrame(chunks)
        if os.path.exists(chunks_path):
            try:
                # Check if file is valid and not empty
                file_size = os.path.getsize(chunks_path)
                if file_size == 0:
                    # File is corrupted/empty - overwrite it
                    df.to_parquet(chunks_path, engine=PARQUET_ENGINE, index=False)
                else:
                    # File is valid - try to append
                    existing = pd.read_parquet(chunks_path, engine=PARQUET_ENGINE)
                    
                    # Remove duplicates by chunk_id (keep newer ones)
                    existing_ids = set(existing["chunk_id"].unique())
                    new_rows = df[~df["chunk_id"].isin(existing_ids)]
                    
                    if len(new_rows) > 0:
                        combined = pd.concat([existing, new_rows], ignore_index=True)
                        combined.to_parquet(chunks_path, engine=PARQUET_ENGINE, index=False)
                        print(f"   ✅ Appended {len(new_rows)} new chunks")
                    else:
                        print(f"   ℹ️  All chunks already in parquet, no new chunks to add")
            except Exception as e:
                # If any error reading/appending, log and FAIL rather than overwrite
                print(f"\n   ❌ ERROR: Could not append to existing parquet file: {e}")
                print(f"   📋 File: {chunks_path}")
                print(f"   💡 TIP: If file is corrupted, delete it and re-run the pipeline")
                raise  # Re-raise error instead of silently overwriting
        else:
            df.to_parquet(chunks_path, engine=PARQUET_ENGINE, index=False)
    else:
        mode = "ab" if os.path.exists(chunks_path) else "wb"
        with open(chunks_path, mode) as f:
            pickle.dump(chunks, f)


# step3_chunk.py — updated chunk_all_documents()

from registry import (
    is_already_processed,
    mark_as_processed,
    mark_as_failed,
    get_next_chunk_id,
    advance_chunk_counter
)

def chunk_all_documents():
    json_files = [
        f for f in os.listdir(EXTRACTED_DIR)
        if f.endswith(".json")
    ]

    # ── Separate new vs already done ──
    new_files  = []
    skip_files = []

    for jf in json_files:
        pdf_name = jf.replace(".json", ".pdf")
        if is_already_processed(pdf_name):
            skip_files.append(pdf_name)
        else:
            new_files.append(jf)

    print(f"\n⏭️  Skipping {len(skip_files)} "
          f"already chunked files:")
    for f in skip_files:
        print(f"   • {f}")

    print(f"\n🆕 New files to chunk: {len(new_files)}")
    for f in new_files:
        print(f"   • {f}")

    if not new_files:
        print("\n✅ Nothing new to chunk!")
        return 0

    chunks_path = ALL_CHUNKS_FILE
    if os.path.exists(chunks_path):
        print("\n📦 Existing chunk store found; appending new chunks.")
    else:
        print("\n📦 Starting fresh chunk store")

    new_chunks_added = 0

    for json_file in new_files:
        pdf_name = json_file.replace(".json", ".pdf")
        meta     = SOURCE_METADATA.get(pdf_name, {})

        print(f"\n✂️  Chunking: {pdf_name}")

        try:
            json_path = os.path.join(EXTRACTED_DIR, json_file)
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Build full text
            full_text = ""
            for page in data["pages"]:
                if page["text"]:
                    full_text += page["text"] + "\n"
                for tbl in page.get("tables_text", []):
                    if tbl:
                        full_text += tbl + "\n"

            # HDR chunk it
            chunks = hdr_chunk(full_text)

            # Get unique starting ID for this batch
            start_id = get_next_chunk_id()
            chunk_ids = []
            chunk_docs = []

            for idx, chunk_text in enumerate(
                tqdm(chunks, desc=f"  Saving")
            ):
                cid          = f"chunk_{start_id + idx:06d}"
                trust_weight = (
                    meta.get("authority_score", 0.8) *
                    (1 - meta.get("bias_penalty", 0.0))
                )

                chunk_doc = {
                    "chunk_id"       : cid,
                    "text"           : chunk_text,
                    "char_count"     : len(chunk_text),
                    "source_file"    : pdf_name,
                    "title"          : meta.get("title", pdf_name),
                    "source_org"     : meta.get("source", ""),
                    "region"         : meta.get("region", "India"),
                    "domain"         : meta.get("domain", "agriculture"),
                    "year"           : meta.get("year", 2023),
                    "authority_score": meta.get("authority_score", 0.8),
                    "bias_penalty"   : meta.get("bias_penalty", 0.0),
                    "trust_weight"   : round(trust_weight, 4),
                    "crop_tags"      : extract_crop_tags(chunk_text),
                }
                chunk_docs.append(chunk_doc)
                chunk_ids.append(cid)

            if chunk_docs:
                save_chunks_batch(chunk_docs, chunks_path)

            # Advance the global counter
            advance_chunk_counter(len(chunks))

            # Register this file as done
            mark_as_processed(pdf_name, len(chunks), chunk_ids)
            new_chunks_added += len(chunks)

        except Exception as e:
            mark_as_failed(pdf_name, str(e))
            print(f"   ❌ Error chunking {pdf_name}: {e}")
            continue

    print(f"\n✅ Added {new_chunks_added} new chunks")
    print("   Chunk store updated at: {}".format(chunks_path))
    return new_chunks_added


if __name__ == "__main__":
    chunk_all_documents()