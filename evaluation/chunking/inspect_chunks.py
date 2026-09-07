"""
inspect_chunks.py
==============================================================================
STAGE 1: CHUNKING INSPECTION & QUALITY EVALUATION UTILITY

Purpose:
  Inspect, sample, and extract chunks from the existing chunk store
  (data/chunks/all_chunks.parquet) for human evaluation.
  DO NOT modify production pipeline files or the parquet file.
==============================================================================
"""

import os
import sys
import argparse
import pandas as pd

# Ensure Windows stdout handles utf-8 gracefully
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

DEFAULT_CHUNKS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "chunks", "all_chunks.parquet")
)
DEFAULT_OUTPUT_CSV = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "chunk_quality_results.csv")
)


def load_chunks(parquet_path=DEFAULT_CHUNKS_PATH) -> pd.DataFrame:
    """Safely loads the chunks parquet file."""
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Chunks parquet file not found at: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    return df


def print_corpus_stats(df: pd.DataFrame):
    """Prints high-level summary statistics of the chunk corpus."""
    print("=" * 80)
    print(" CHUNK CORPUS SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total Chunks:           {len(df):,}")
    print(f"Unique Sources (PDFs):  {df['source_file'].nunique()}")
    print(f"Chunk Length (Chars):")
    print(f"  - Min:    {df['char_count'].min():,} chars")
    print(f"  - Max:    {df['char_count'].max():,} chars")
    print(f"  - Mean:   {df['char_count'].mean():.1f} chars")
    print(f"  - Median: {df['char_count'].median():.1f} chars")
    print(f"  - Std:    {df['char_count'].std():.1f} chars")
    print("-" * 80)


def list_sources(df: pd.DataFrame):
    """Lists unique source documents and their chunk counts."""
    print("\n[SOURCES] Unique Source Documents:")
    counts = df["source_file"].value_counts()
    for idx, (source, count) in enumerate(counts.items(), 1):
        print(f"  {idx:2d}. {source} ({count:,} chunks)")


def list_crops(df: pd.DataFrame):
    """Lists crop tags found in the chunks."""
    all_crops = []
    for tags in df["crop_tags"]:
        if isinstance(tags, (list, tuple)):
            all_crops.extend(tags)
        elif hasattr(tags, "tolist"):
            all_crops.extend(tags.tolist())
    series = pd.Series(all_crops).value_counts()
    print("\n[CROPS] Detected Crop Tags:")
    for crop, count in series.head(20).items():
        print(f"  - {crop:<15}: {count:,} chunks")


def format_chunk(row: pd.Series, index: int = 1) -> str:
    """Formats a single chunk record for terminal display."""
    crop_str = ", ".join(row['crop_tags']) if len(row['crop_tags']) > 0 else "None detected"
    source = row.get('source_file', 'Unknown')
    chunk_id = row.get('chunk_id', 'Unknown')
    char_count = row.get('char_count', len(row.get('text', '')))
    trust = row.get('trust_weight', 'N/A')
    
    separator = "-" * 80
    header = f"[{index}] CHUNK ID: {chunk_id} | SOURCE: {source} | LENGTH: {char_count} chars"
    meta = f"    Crop Tags: [{crop_str}] | Trust Weight: {trust} | Page: N/A (untracked in parquet)"
    text = row.get('text', '').strip()

    return f"{separator}\n{header}\n{meta}\n{separator}\n{text}\n"


def export_sample_to_csv(sample_df: pd.DataFrame, output_csv: str = DEFAULT_OUTPUT_CSV, append: bool = False):
    """
    Exports sampled chunks to a CSV formatted specifically for human inspection.
    """
    records = []
    for _, row in sample_df.iterrows():
        crops = ", ".join(row['crop_tags']) if len(row['crop_tags']) > 0 else "None"
        records.append({
            "chunk_id": row["chunk_id"],
            "source": row["source_file"],
            "page": "N/A",  # Not stored in production parquet
            "crop": crops,
            "chunk_text": row["text"].replace("\r", " ").replace("\n", " "),
            "chunk_length": row["char_count"],
            "context_complete": "",           # Human evaluation: 1 (Complete) / 0 (Fragmented)
            "topic_coherent": "",            # Human evaluation: 1 (Single topic) / 0 (Mixed unrelated)
            "contains_useful_information": "",# Human evaluation: 1 (Actionable agri info) / 0 (Boilerplate/junk)
            "table_information_preserved": "",# Human evaluation: 1 (Yes) / 0 (Corrupted table) / NA (No table)
            "human_quality_score": "",        # Human evaluation: 1 (Poor) to 5 (Excellent)
            "notes": ""                       # Human evaluation notes
        })

    out_df = pd.DataFrame(records)

    if append and os.path.exists(output_csv):
        existing = pd.read_csv(output_csv)
        combined = pd.concat([existing, out_df]).drop_duplicates(subset=["chunk_id"])
        combined.to_csv(output_csv, index=False)
        print(f"\n[OK] Appended {len(out_df)} sample chunks to: {output_csv}")
    else:
        out_df.to_csv(output_csv, index=False)
        print(f"\n[OK] Exported {len(out_df)} sample chunks for human evaluation to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect existing RAG chunks for Stage 1 Chunking Evaluation."
    )
    parser.add_argument("--parquet", type=str, default=DEFAULT_CHUNKS_PATH, help="Path to all_chunks.parquet")
    parser.add_argument("--stats", action="store_true", help="Display chunk corpus summary statistics")
    parser.add_argument("--list-sources", action="store_true", help="List all source PDFs and chunk counts")
    parser.add_argument("--list-crops", action="store_true", help="List detected crops and counts")
    parser.add_argument("--sample", type=int, default=0, help="Randomly sample N chunks to inspect")
    parser.add_argument("--source", type=str, default=None, help="Filter chunks by source PDF filename substring")
    parser.add_argument("--crop", type=str, default=None, help="Filter chunks mentioning a specific crop")
    parser.add_argument("--shortest", type=int, default=0, help="Inspect N shortest chunks (outlier analysis)")
    parser.add_argument("--longest", type=int, default=0, help="Inspect N longest chunks (outlier analysis)")
    parser.add_argument("--chunk-id", type=str, default=None, help="Inspect a specific chunk by chunk_id")
    parser.add_argument("--export-csv", type=str, default=None, help="Export inspected chunks to CSV for human evaluation")
    parser.add_argument("--append-csv", action="store_true", help="Append to existing CSV rather than overwrite")

    args = parser.parse_args()

    # Load dataset
    df = load_chunks(args.parquet)

    # If no filtering or inspection argument is provided, show stats and 3 sample chunks
    has_action = any([
        args.stats, args.list_sources, args.list_crops, args.sample > 0,
        args.source, args.crop, args.shortest > 0, args.longest > 0,
        args.chunk_id, args.export_csv
    ])

    if not has_action:
        print_corpus_stats(df)
        print("\n[TIP] Run `python inspect_chunks.py --help` to see sampling and filter options.")
        print("\nShowing 3 random chunks for initial inspection:")
        sample = df.sample(3, random_state=42)
        for idx, (_, row) in enumerate(sample.iterrows(), 1):
            print(format_chunk(row, idx))
        return

    if args.stats:
        print_corpus_stats(df)

    if args.list_sources:
        list_sources(df)

    if args.list_crops:
        list_crops(df)

    # Filter pipeline
    subset = df.copy()

    if args.source:
        subset = subset[subset["source_file"].str.contains(args.source, case=False, na=False)]
        print(f"\n[FILTER] Filtered by source '{args.source}': {len(subset):,} matching chunks.")

    if args.crop:
        target_crop = args.crop.lower()
        subset = subset[subset["crop_tags"].apply(lambda tags: target_crop in [t.lower() for t in tags])]
        print(f"\n[FILTER] Filtered by crop '{args.crop}': {len(subset):,} matching chunks.")

    if args.chunk_id:
        subset = subset[subset["chunk_id"] == args.chunk_id]
        print(f"\n[FILTER] Filtered by chunk_id '{args.chunk_id}': {len(subset)} matching chunks.")

    # Ordering / Outliers / Sampling
    result_df = pd.DataFrame()

    if args.shortest > 0:
        result_df = subset.sort_values(by="char_count", ascending=True).head(args.shortest)
        print(f"\n[OUTLIER] Displaying {len(result_df)} SHORTEST chunks:")
    elif args.longest > 0:
        result_df = subset.sort_values(by="char_count", ascending=False).head(args.longest)
        print(f"\n[OUTLIER] Displaying {len(result_df)} LONGEST chunks:")
    elif args.sample > 0:
        n = min(args.sample, len(subset))
        result_df = subset.sample(n, random_state=42)
        print(f"\n[SAMPLE] Random sample of {n} chunks:")
    elif args.chunk_id or args.source or args.crop:
        # Default to first 5 if filtered without explicit sample size
        result_df = subset.head(5)

    if not result_df.empty:
        for idx, (_, row) in enumerate(result_df.iterrows(), 1):
            print(format_chunk(row, idx))

        if args.export_csv:
            export_sample_to_csv(result_df, output_csv=args.export_csv, append=args.append_csv)


if __name__ == "__main__":
    main()
