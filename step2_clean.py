# step2_clean.py
# =============================================
# PHASE 4: Clean all extracted text
# =============================================

import re
import os
import json
from tqdm import tqdm
from config import EXTRACTED_DIR


# ── Noise patterns common in Indian agricultural PDFs ──
NOISE_PATTERNS = [
    # University headers/footers (repeated on every page)
    r'Tamil Nadu Agricultural University.*?\n',
    r'Kerala Agricultural University.*?\n',
    r'TNAU.*?Coimbatore.*?\n',
    r'National Centre for Organic.*?\n',
    r'Package of Practices.*?\n',
    r'Organic Package.*?Crops\s*\n',

    # Page numbers (standalone digits)
    r'^\s*\d{1,3}\s*$',

    # Repeated copyright/source lines
    r'Source\s*:.*?\n',
    r'Published by.*?\n',

    # Form feed characters
    r'\x0c',

    # Excessive dots (table of contents lines)
    r'\.{4,}',
]


def clean_text(text):
    """
    Clean agricultural PDF text in 6 steps.
    """
    if not text:
        return ""

    # Step 1: Remove noise patterns
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, ' ', text,
                      flags=re.MULTILINE | re.IGNORECASE)

    # Step 2: Fix hyphenated line breaks
    # "irri-\nigation" → "irrigation"
    text = re.sub(r'-\s*\n\s*', '', text)

    # Step 3: Fix spacing around agricultural measurements
    # "kg /ha" → "kg/ha", "t/ ha" → "t/ha"
    text = re.sub(r'(\w+)\s*/\s*(\w+)', r'\1/\2', text)

    # Step 4: Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)       # multiple spaces
    text = re.sub(r'\n{3,}', '\n\n', text)    # max 2 newlines

    # Step 5: Fix common OCR errors in agricultural text
    ocr_fixes = {
        'ha\\.': 'ha.',        # "ha." misread
        '@ ':    'at ',        # "@" in dosage instructions
        'Oo ':   '0 ',         # OCR zero/letter confusion
    }
    for wrong, right in ocr_fixes.items():
        text = text.replace(wrong, right)

    # Step 6: Final strip
    return text.strip()


def clean_all_extracted():
    """
    Clean all extracted JSON files.
    Overwrites them with cleaned versions.
    """
    json_files = [
        f for f in os.listdir(EXTRACTED_DIR)
        if f.endswith(".json")
    ]

    if not json_files:
        print("❌ No extracted JSON files found.")
        print("   Run step1_extract.py first!")
        return

    print(f"\n🧹 Cleaning {len(json_files)} extracted files...")

    for json_file in tqdm(json_files, desc="Cleaning"):
        json_path = os.path.join(EXTRACTED_DIR, json_file)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        total_before = 0
        total_after  = 0

        for page in data["pages"]:
            before = len(page["text"])
            page["text"] = clean_text(page["text"])
            after = len(page["text"])

            total_before += before
            total_after  += after

            # Clean table text too
            page["tables_text"] = [
                clean_text(t) for t in page["tables_text"]
            ]

        # Add cleaning stats
        data["cleaning_stats"] = {
            "chars_before": total_before,
            "chars_after":  total_after,
            "chars_removed": total_before - total_after,
            "reduction_pct": round(
                (total_before - total_after) /
                max(total_before, 1) * 100, 1
            )
        }

        # Save cleaned version
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"   ✅ {json_file}: "
              f"removed {data['cleaning_stats']['chars_removed']:,} "
              f"noise chars "
              f"({data['cleaning_stats']['reduction_pct']}%)")

    print("\n✅ All files cleaned!")


if __name__ == "__main__":
    clean_all_extracted()