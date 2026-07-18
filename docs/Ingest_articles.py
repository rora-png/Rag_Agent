"""
Pulls articles from URLs, extracts clean main-body text (no nav/ads/footers),
and saves each as a .txt file ready for chunking + embedding in your RAG pipeline.
"""

import trafilatura
import os
import re
from urllib.parse import urlparse

# List the article URLs you want to ingest here
URLS = [
    "https://en.wikipedia.org/wiki/BERT_(language_model)",
    "https://aws.amazon.com/what-is/retrieval-augmented-generation/",
    "https://en.wikipedia.org/wiki/Docker_(software)",
    # add more URLs
]

OUTPUT_DIR = "articles"


def slugify(url: str) -> str:
    """Turn a URL into a safe filename."""
    path = urlparse(url).path.strip("/")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", path or urlparse(url).netloc)
    return slug.strip("_").lower()[:80] or "article"


def fetch_and_clean(url: str) -> dict | None:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        print(f"  [FAILED] Could not fetch: {url}")
        return None

    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    )
    metadata = trafilatura.extract_metadata(downloaded)

    if not text:
        print(f"  [FAILED] Could not extract main content: {url}")
        return None

    return {
        "url": url,
        "title": metadata.title if metadata else "",
        "author": metadata.author if metadata else "",
        "date": metadata.date if metadata else "",
        "text": text,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for url in URLS:
        print(f"Fetching: {url}")
        result = fetch_and_clean(url)
        if not result:
            continue

        filename = slugify(url) + ".txt"
        filepath = os.path.join(OUTPUT_DIR, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Title: {result['title']}\n")
            f.write(f"Author: {result['author']}\n")
            f.write(f"Date: {result['date']}\n")
            f.write(f"Source: {result['url']}\n")
            f.write("\n---\n\n")
            f.write(result["text"])

        print(f"  [OK] Saved -> {filepath}")

    print(f"\nDone. {len(URLS)} URL(s) processed. Files saved in ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()