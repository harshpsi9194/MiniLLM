# SciMiniLM Scientific Text Dataset

This directory contains a curated multi-domain scientific-text corpus formatted as JSON Lines (`.jsonl`) with a `"text"` field compatible with `sciminilm.training.dataset.iter_text_files` and `scripts/train_500m.py`.

## Dataset Statistics

| Split | File | Records | Size (MB) |
| --- | --- | ---: | ---: |
| Train | `data/train.jsonl` | 5,446 | 8.84 |
| Validation | `data/val.jsonl` | 286 | 0.47 |
| **Total** | — | **5,732** | **9.32** |

- **Total characters:** 8,517,145
- **Total whitespace words:** 1,151,490
- **Approximate BPE tokens (~1.35 tok/word):** ~1,554,511
- **Sources:** arXiv Atom API (5,732 papers)

### Domain Breakdown

| Domain | Records |
| --- | ---: |
| Machine Learning | 450 |
| Cosmology and Extragalactic Astrophysics | 449 |
| Computational Physics | 445 |
| Quantum Physics | 443 |
| Quantitative Methods in Biology | 438 |
| Probability | 428 |
| Biomolecules | 425 |
| Optimization and Control | 423 |
| Materials Science | 412 |
| Computation and Language | 402 |
| High Energy Physics - Theory | 396 |
| Chemical Physics | 395 |
| Statistical Machine Learning | 357 |
| Artificial Intelligence | 269 |

## Provenance & Licensing

- **arXiv** (`https://export.arxiv.org/api/query`): paper metadata and abstracts retrieved via the public arXiv API (arXiv metadata is released under CC0 1.0 Public Domain Dedication).
- **EuropePMC** (`https://www.ebi.ac.uk/europepmc/webservices/rest/search`): open biomedical and life-science abstracts retrieved via the public EuropePMC REST API.

## Usage with SciMiniLM

Stream directly during training:

```bash
python scripts/train_500m.py --data-files data/train.jsonl --max-steps 50
```

## Scaling to 500M Tokens

This repository ships a GitHub-friendly ~9.3 MB multi-domain sample for reproducible local runs and pipeline verification. To scale up to the full **500M-token** pretraining budget:

1. Increase fetch counts with `scripts/download_dataset.py --arxiv-per-cat 5000 --epmc-per-query 5000`, or point `iter_text_files` / `PackedTokenStreamingDataset` at larger scientific corpora (e.g., `allenai/peS2o`, arXiv/PubMed Central Open Access subsets).
2. Pre-tokenize text shards into flat binary `.bin` memory-mapped files via `sciminilm.training.dataset.write_token_shard` and pass `--bin-shard` / `--val-bin-shard` to `scripts/train_500m.py`.
