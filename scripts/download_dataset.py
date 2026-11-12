"""Download a multi-domain scientific-text dataset (arXiv + EuropePMC) into data/."""

from __future__ import annotations

import argparse
import html
import json
import random
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ARXIV_CATEGORIES: Tuple[Tuple[str, str], ...] = (
    ("cs.LG", "Machine Learning"),
    ("cs.CL", "Computation and Language"),
    ("cs.AI", "Artificial Intelligence"),
    ("quant-ph", "Quantum Physics"),
    ("physics.comp-ph", "Computational Physics"),
    ("physics.chem-ph", "Chemical Physics"),
    ("cond-mat.mtrl-sci", "Materials Science"),
    ("astro-ph.CO", "Cosmology and Extragalactic Astrophysics"),
    ("hep-th", "High Energy Physics - Theory"),
    ("math.OC", "Optimization and Control"),
    ("math.PR", "Probability"),
    ("stat.ML", "Statistical Machine Learning"),
    ("q-bio.QM", "Quantitative Methods in Biology"),
    ("q-bio.BM", "Biomolecules"),
)

EUROPEPMC_QUERIES: Tuple[Tuple[str, str], ...] = (
    ("molecular biology AND HAS_ABSTRACT:Y", "Molecular Biology"),
    ("organic chemistry synthesis AND HAS_ABSTRACT:Y", "Chemistry"),
    ("genomics protein folding AND HAS_ABSTRACT:Y", "Genomics & Biophysics"),
)

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_arxiv_records(
    category: str,
    domain_label: str,
    max_results: int = 500,
    batch_size: int = 250,
    delay_s: float = 0.5,
) -> List[Dict[str, str]]:
    """Fetch paper titles and abstracts for an arXiv category via the Atom API."""
    records: List[Dict[str, str]] = []
    start = 0
    while start < max_results:
        cur_max = min(batch_size, max_results - start)
        params = urllib.parse.urlencode(
            {
                "search_query": f"cat:{category}",
                "start": start,
                "max_results": cur_max,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
        )
        url = f"https://export.arxiv.org/api/query?{params}"
        req = urllib.request.Request(
            url, headers={"User-Agent": "SciMiniLM-DatasetFetcher/0.1 (EE656)"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            xml_bytes = resp.read()

        root = ET.fromstring(xml_bytes)
        entries = root.findall("atom:entry", ATOM_NS)
        if not entries:
            break

        for entry in entries:
            raw_id = entry.findtext("atom:id", default="", namespaces=ATOM_NS)
            raw_title = entry.findtext("atom:title", default="", namespaces=ATOM_NS)
            raw_summary = entry.findtext("atom:summary", default="", namespaces=ATOM_NS)
            title = _clean_text(raw_title)
            abstract = _clean_text(raw_summary)
            if len(abstract) < 80 or not title:
                continue
            records.append(
                {
                    "id": raw_id.rsplit("/", 1)[-1] if raw_id else "",
                    "source": "arxiv",
                    "domain": domain_label,
                    "category": category,
                    "title": title,
                    "text": f"Title: {title}\nDomain: {domain_label} ({category})\n\n{abstract}",
                }
            )

        start += len(entries)
        if len(entries) < cur_max:
            break
        if delay_s > 0:
            time.sleep(delay_s)

    return records


def fetch_europepmc_records(
    query: str,
    domain_label: str,
    max_results: int = 200,
    page_size: int = 50,
    delay_s: float = 0.1,
) -> List[Dict[str, str]]:
    """Fetch scientific abstracts from the EuropePMC open REST API."""
    records: List[Dict[str, str]] = []
    cursor = "*"
    while len(records) < max_results:
        cur_size = min(page_size, max_results - len(records))
        params = urllib.parse.urlencode(
            {
                "query": query,
                "resultType": "core",
                "cursorMark": cursor,
                "pageSize": cur_size,
                "format": "json",
            }
        )
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?{params}"
        req = urllib.request.Request(
            url, headers={"User-Agent": "SciMiniLM-DatasetFetcher/0.1 (EE656)"}
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            print(f"  [warn] EuropePMC request stopped early: {exc}")
            break

        items = payload.get("resultList", {}).get("result", [])
        if not items:
            break

        for item in items:
            title = _clean_text(item.get("title") or "")
            abstract = _clean_text(item.get("abstractText") or "")
            pmid = str(item.get("pmid") or item.get("id") or "")
            if len(abstract) < 80 or not title:
                continue
            records.append(
                {
                    "id": pmid,
                    "source": "europepmc",
                    "domain": domain_label,
                    "category": domain_label,
                    "title": title,
                    "text": f"Title: {title}\nDomain: {domain_label}\n\n{abstract}",
                }
            )

        next_cursor = payload.get("nextCursorMark")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
        if delay_s > 0:
            time.sleep(delay_s)

    return records


def write_jsonl(records: List[Dict[str, str]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path.stat().st_size


def write_data_readme(
    out_dir: Path,
    train_records: List[Dict[str, str]],
    val_records: List[Dict[str, str]],
    train_bytes: int,
    val_bytes: int,
) -> None:
    all_records = train_records + val_records
    source_counts = Counter(r["source"] for r in all_records)
    domain_counts = Counter(r["domain"] for r in all_records)
    total_chars = sum(len(r["text"]) for r in all_records)
    total_words = sum(len(r["text"].split()) for r in all_records)
    est_tokens = int(total_words * 1.35)

    source_parts = []
    if source_counts.get("arxiv", 0) > 0:
        source_parts.append(f"arXiv Atom API ({source_counts['arxiv']:,} papers)")
    if source_counts.get("europepmc", 0) > 0:
        source_parts.append(f"EuropePMC REST API ({source_counts['europepmc']:,} papers)")
    sources_str = ", ".join(source_parts) if source_parts else "None"

    domain_rows = "\n".join(
        f"| {dom} | {cnt:,} |" for dom, cnt in domain_counts.most_common()
    )
    readme = f"""# SciMiniLM Scientific Text Dataset

This directory contains a curated multi-domain scientific-text corpus formatted as JSON Lines (`.jsonl`) with a `"text"` field compatible with `sciminilm.training.dataset.iter_text_files` and `scripts/train_500m.py`.

## Dataset Statistics

| Split | File | Records | Size (MB) |
| --- | --- | ---: | ---: |
| Train | `data/train.jsonl` | {len(train_records):,} | {train_bytes / (1024 * 1024):.2f} |
| Validation | `data/val.jsonl` | {len(val_records):,} | {val_bytes / (1024 * 1024):.2f} |
| **Total** | — | **{len(all_records):,}** | **{(train_bytes + val_bytes) / (1024 * 1024):.2f}** |

- **Total characters:** {total_chars:,}
- **Total whitespace words:** {total_words:,}
- **Approximate BPE tokens (~1.35 tok/word):** ~{est_tokens:,}
- **Sources:** {sources_str}

### Domain Breakdown

| Domain | Records |
| --- | ---: |
{domain_rows}

## Provenance & Licensing

- **arXiv** (`https://export.arxiv.org/api/query`): paper metadata and abstracts retrieved via the public arXiv API (arXiv metadata is released under CC0 1.0 Public Domain Dedication).
- **EuropePMC** (`https://www.ebi.ac.uk/europepmc/webservices/rest/search`): open biomedical and life-science abstracts retrieved via the public EuropePMC REST API.

## Usage with SciMiniLM

Stream directly during training:

```bash
python scripts/train_500m.py --data-files data/train.jsonl --max-steps 50
```

## Scaling to 500M Tokens

This repository ships a GitHub-friendly ~{(train_bytes + val_bytes) / (1024 * 1024):.1f} MB multi-domain sample for reproducible local runs and pipeline verification. To scale up to the full **500M-token** pretraining budget:

1. Increase fetch counts with `scripts/download_dataset.py --arxiv-per-cat 5000 --epmc-per-query 5000`, or point `iter_text_files` / `PackedTokenStreamingDataset` at larger scientific corpora (e.g., `allenai/peS2o`, arXiv/PubMed Central Open Access subsets).
2. Pre-tokenize text shards into flat binary `.bin` memory-mapped files via `sciminilm.training.dataset.write_token_shard` and pass `--bin-shard` / `--val-bin-shard` to `scripts/train_500m.py`.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download multi-domain scientific text dataset into data/train.jsonl and data/val.jsonl."
    )
    parser.add_argument("--out-dir", type=str, default="data")
    parser.add_argument("--arxiv-per-cat", type=int, default=450)
    parser.add_argument("--epmc-per-query", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    all_records: List[Dict[str, str]] = []

    for cat, label in ARXIV_CATEGORIES:
        print(f"Fetching arXiv [{cat}] ({label})...")
        recs = fetch_arxiv_records(cat, label, max_results=args.arxiv_per_cat)
        print(f"  -> {len(recs)} records")
        all_records.extend(recs)

    if args.epmc_per_query > 0:
        for query, label in EUROPEPMC_QUERIES:
            print(f"Fetching EuropePMC ({label})...")
            recs = fetch_europepmc_records(query, label, max_results=args.epmc_per_query)
            print(f"  -> {len(recs)} records")
            all_records.extend(recs)

    # Deduplicate by normalized title
    seen_titles = set()
    deduped: List[Dict[str, str]] = []
    for rec in all_records:
        key = rec["title"].lower()
        if key not in seen_titles:
            seen_titles.add(key)
            deduped.append(rec)

    if len(deduped) < 20:
        raise RuntimeError(f"Expected >=20 downloaded records, got {len(deduped)}")

    rng = random.Random(args.seed)
    rng.shuffle(deduped)

    val_count = max(1, int(len(deduped) * args.val_fraction))
    val_records = deduped[:val_count]
    train_records = deduped[val_count:]

    train_bytes = write_jsonl(train_records, out_dir / "train.jsonl")
    val_bytes = write_jsonl(val_records, out_dir / "val.jsonl")
    write_data_readme(out_dir, train_records, val_records, train_bytes, val_bytes)

    print(
        f"Saved {len(train_records):,} train ({train_bytes / (1024*1024):.2f} MB) and "
        f"{len(val_records):,} val ({val_bytes / (1024*1024):.2f} MB) records to {out_dir}/"
    )


if __name__ == "__main__":
    main()
