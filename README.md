# SciMiniLM

**EE656 Course Project (Prof. Nishchal K. Verma)** — A clean, modular PyTorch implementation of a **77M-parameter** decoder-only Transformer for scientific text modeling, featuring configurable MHA/GQA/MQA attention with FlashAttention/SDPA, INT8/INT4 weight quantization, KV-cached & speculative decoding, a custom Triton RMSNorm kernel (with PyTorch fallback), and a 500M-token training pipeline.

## Architecture (77M Preset)

`ModelConfig.sciminilm_77m()` configures:
- `vocab_size = 32000` (tied input/output embeddings)
- `d_model = 640`, `n_layers = 12`, `n_heads = 10` (`head_dim = 64`)
- `n_kv_heads`: `10` (MHA), `2` (GQA), or `1` (MQA)
- `d_ff = 1600` (SwiGLU MLP)
- `max_seq_len = 2048`, RoPE (`theta = 10000.0`), RMSNorm (`eps = 1e-6`)
- Exact unique parameter count (MHA): **77,020,800** (~77.02M)

## Package Structure

- `sciminilm/config.py`: `ModelConfig`, `QuantConfig`, `InferenceConfig`, `TrainConfig`
- `sciminilm/layers/`:
  - `rmsnorm.py`: Custom Triton CUDA forward/backward RMSNorm kernel with graceful PyTorch fallback
  - `rope.py`: Rotary Position Embeddings with offset/KV-cache support
  - `attention.py`: Configurable MHA/GQA/MQA causal attention using `F.scaled_dot_product_attention` (FlashAttention) and manual fallback
  - `mlp.py`: SwiGLU feed-forward network
- `sciminilm/model.py`: `TransformerBlock` and `SciMiniLM`
- `sciminilm/quant/`: INT8 per-channel & INT4 group-wise (4-bit packed `uint8`) quantization, `QuantizedLinear`, memory measurement, and perplexity evaluation harness
- `sciminilm/inference/`: `KVCache` with rollback, Top-K/Top-P sampling, `InferenceEngine`, and `SpeculativeDecoder`
- `sciminilm/training/`: Byte-level BPE `ScientificTokenizer`, `PackedTokenStreamingDataset` & `MemmapTokenDataset` (500M token budget), and `Trainer`
- `sciminilm/benchmarks/`: MHA vs GQA vs MQA comparison across context lengths on perplexity, KV-cache memory, and latency
- `data/`: Multi-domain scientific-text corpus (`train.jsonl`, `val.jsonl`, `README.md`) fetched from arXiv & EuropePMC

## Running Tests & Scripts

```bash
source .venv/bin/activate
pytest
```

CLI tools in `scripts/`:
- `python scripts/download_dataset.py --out-dir data`
- `python scripts/train_500m.py --data-files data/train.jsonl --max-steps 50`
- `python scripts/train_500m.py --dry-run`
- `python scripts/bench_attention.py --preset tiny`
- `python scripts/eval_quant.py --preset tiny`
- `python scripts/generate.py --speculative`
