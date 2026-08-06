# gov_paper_fetcher.py
# =============================================
# Fetches research papers and PDFs from
# Indian .gov and .nic.in agricultural websites
# Feeds them into your existing pipeline
# =============================================

import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from config import RAW_PDF_DIR

# ── Target .gov sources ──
GOV_SOURCES = [
    {
        "name"    : "ICAR Krishi Repository",
        "url"     : "https://krishi.icar.gov.in/jspui/",
        "type"    : "repository",
        "domain"  : "icar.gov.in",
        "priority": 1.0,
    },
    {
        "name"    : "NIPHM Publications",
        "url"     : "https://niphm.gov.in/Publications.html",
        "type"    : "publications_page",
        "domain"  : "niphm.gov.in",
        "priority": 1.0,
    },
    {
        "name"    : "NCOF Publications",
        "url"     : "https://ncof.dacnet.nic.in/Publications.aspx",
        "type"    : "publications_page",
        "domain"  : "ncof.dacnet.nic.in",
        "priority": 1.0,
    },
    {
        "name"    : "Ministry of Agriculture",
        "url"     : "https://agricoop.nic.in/en/publication",
        "type"    : "publications_page",
        "domain"  : "agricoop.nic.in",
        "priority": 1.0,
    },
    {
        "name"    : "DARE Publications",
        "url"     : "https://dare.gov.in/publications",
        "type"    : "publications_page",
        "domain"  : "dare.gov.in",
        "priority": 1.0,
    },
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; "
                  "AgriRAG/1.0; Research purposes; "
                  "contact: your@email.com)"
}


def get_pdf_links(url, source_name):
    """
    Scrape a .gov page and find all PDF links.
    Returns list of (pdf_url, title) tuples.
    """
    print(f"\n🔍 Scanning: {source_name}")
    print(f"   URL: {url}")

    try:
        resp = requests.get(url, headers=HEADERS,
                            timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")

        pdf_links = []
        base_url  = f"{urlparse(url).scheme}://{urlparse(url).netloc}"

        # Find all links ending in .pdf
        for tag in soup.find_all("a", href=True):
            href  = tag["href"]
            title = tag.get_text(strip=True) or "Untitled"

            if ".pdf" in href.lower():
                # Make absolute URL
                if href.startswith("http"):
                    full_url = href
                elif href.startswith("/"):
                    full_url = base_url + href
                else:
                    full_url = urljoin(url, href)

                if title and len(title) > 3:
                    pdf_links.append((full_url, title))

        print(f"   Found: {len(pdf_links)} PDFs")
        return pdf_links

    except Exception as e:
        print(f"   ❌ Error scanning {source_name}: {e}")
        return []


def download_pdf(pdf_url, filename, source_name):
    """
    Download a single PDF to data/raw pdf/ folder.
    Returns True if successful.
    """
    filepath = os.path.join(RAW_PDF_DIR, filename)

    # Skip if already downloaded
    if os.path.exists(filepath):
        print(f"   ⏭️  Already exists: {filename}")
        return False

    try:
        print(f"   ⬇️  Downloading: {filename[:50]}...")
        resp = requests.get(
            pdf_url, headers=HEADERS,
            timeout=30, stream=True
        )
        resp.raise_for_status()

        # Check it's actually a PDF
        content_type = resp.headers.get(
            "content-type", ""
        ).lower()
        if "pdf" not in content_type and \
           not pdf_url.lower().endswith(".pdf"):
            print(f"   ⚠️  Not a PDF: {content_type}")
            return False

        # Save file
        with open(filepath, "wb") as f:
            for chunk in resp.iter_content(
                chunk_size=8192
            ):
                f.write(chunk)

        size_kb = os.path.getsize(filepath) / 1024
        print(f"   ✅ Saved: {filename} "
              f"({size_kb:.0f} KB)")
        return True

    except Exception as e:
        print(f"   ❌ Download failed: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False


def clean_filename(title, source_name, index):
    """
    Convert paper title to safe filename.
    """
    # Remove special characters
    safe = "".join(
        c if c.isalnum() or c in " -_"
        else "_"
        for c in title
    )
    # Shorten
    safe = safe[:60].strip().replace(" ", "_")
    # Add source prefix
    source_short = source_name.split()[0]
    return f"{source_short}_{safe}_{index}.pdf"


def build_metadata_for_gov_pdf(filename, title,
                                source_info):
    """
    Auto-generate config.py metadata entry
    for a downloaded .gov PDF.
    """
    return {
        "title"          : title,
        "source"         : source_info["name"],
        "publisher"      : source_info["domain"],
        "year"           : 2024,
        "authority_score": source_info["priority"],
        "bias_penalty"   : 0.0,
        "domain"         : "agriculture_research",
        "region"         : "India",
    }


def fetch_gov_papers(
    max_per_source=10,
    sources=None
):
    """
    Main function: scrape and download PDFs
    from all configured .gov sources.

    Args:
        max_per_source: max PDFs to download per site
        sources: list of source dicts (default: all)

    Returns:
        downloaded: list of (filename, metadata) tuples
    """
    sources     = sources or GOV_SOURCES
    downloaded  = []
    metadata_new = {}

    print("\n" + "="*55)
    print("  📄 GOV PAPER FETCHER")
    print("="*55)
    print(f"  Sources     : {len(sources)}")
    print(f"  Max per site: {max_per_source}")
    print(f"  Output dir  : {RAW_PDF_DIR}")

    for source in sources:
        print(f"\n{'─'*55}")
        print(f"  Source: {source['name']}")

        # Get PDF links
        pdf_links = get_pdf_links(
            source["url"], source["name"]
        )

        if not pdf_links:
            print(f"  No PDFs found — skipping")
            continue

        count = 0
        for i, (pdf_url, title) in enumerate(pdf_links):
            if count >= max_per_source:
                break

            filename = clean_filename(
                title, source["name"], i
            )

            success = download_pdf(
                pdf_url, filename, source["name"]
            )

            if success:
                meta = build_metadata_for_gov_pdf(
                    filename, title, source
                )
                downloaded.append((filename, meta))
                metadata_new[filename] = meta
                count += 1
                time.sleep(1)  # polite delay

        print(f"\n  Downloaded {count} PDFs "
              f"from {source['name']}")

    # Print metadata to add to config.py
    if metadata_new:
        print(f"\n{'='*55}")
        print("  📋 ADD THIS TO config.py → SOURCE_METADATA:")
        print(f"{'='*55}")
        for fname, meta in metadata_new.items():
            print(f'\n  "{fname}": {{')
            for k, v in meta.items():
                val = f'"{v}"' if isinstance(v,str) else v
                print(f'    "{k}": {val},')
            print(f"  }},")

    print(f"\n{'='*55}")
    print(f"  ✅ Total downloaded: {len(downloaded)} PDFs")
    print(f"  Next step: run add_new_data.py")
    print(f"{'='*55}")

    return downloaded


if __name__ == "__main__":
    fetch_gov_papers(max_per_source=5)