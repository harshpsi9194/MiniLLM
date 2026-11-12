"""Configuration dataclasses and presets for SciMiniLM."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Optional


AttentionType = Literal["mha", "gqa", "mqa"]


@dataclass
class ModelConfig:
    """Configuration for the SciMiniLM decoder-only Transformer.

    Attributes:
        vocab_size: Number of tokens in vocabulary.
        d_model: Hidden dimension of the Transformer residual stream.
        n_layers: Number of Transformer decoder blocks.
        n_heads: Number of query attention heads.
        n_kv_heads: Number of key/value heads. Defaults to n_heads (MHA) if None.
            Set to 1 for MQA, or a divisor of n_heads (1 < n_kv_heads < n_heads) for GQA.
        d_ff: Intermediate dimension of the SwiGLU MLP.
        max_seq_len: Maximum supported context length for RoPE and causal masking.
        rope_theta: Base frequency for Rotary Position Embeddings.
        rms_norm_eps: Numerical stability epsilon for RMSNorm.
        dropout: Dropout probability applied to attention weights and residual projections.
        tie_word_embeddings: Whether to tie input token embeddings with the LM head weight.
        use_flash_attention: Whether to prefer PyTorch SDPA / FlashAttention kernels.
        use_triton_rmsnorm: Whether to prefer the custom Triton RMSNorm kernel on CUDA.
        bias: Whether to include bias terms in linear layers.
        init_std: Standard deviation for weight initialization.
    """

    vocab_size: int = 32000
    d_model: int = 640
    n_layers: int = 12
    n_heads: int = 10
    n_kv_heads: Optional[int] = None
    d_ff: int = 1600
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    tie_word_embeddings: bool = True
    use_flash_attention: bool = True
    use_triton_rmsnorm: bool = True
    bias: bool = False
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )
        if self.n_kv_heads < 1 or self.n_kv_heads > self.n_heads:
            raise ValueError(
                f"n_kv_heads ({self.n_kv_heads}) must be in [1, n_heads ({self.n_heads})]"
            )
        if self.vocab_size <= 0 or self.n_layers <= 0 or self.d_ff <= 0:
            raise ValueError("vocab_size, n_layers, and d_ff must be positive integers")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def attention_variant(self) -> AttentionType:
        assert self.n_kv_heads is not None
        if self.n_kv_heads == self.n_heads:
            return "mha"
        if self.n_kv_heads == 1:
            return "mqa"
        return "gqa"

    def estimate_num_params(self) -> int:
        """Analytically compute exact parameter count for this configuration."""
        assert self.n_kv_heads is not None
        emb_params = self.vocab_size * self.d_model
        lm_head_params = 0 if self.tie_word_embeddings else self.vocab_size * self.d_model

        q_params = self.d_model * self.d_model
        kv_dim = self.n_kv_heads * self.head_dim
        k_params = self.d_model * kv_dim
        v_params = self.d_model * kv_dim
        o_params = self.d_model * self.d_model
        attn_params = q_params + k_params + v_params + o_params
        if self.bias:
            attn_params += self.d_model + 2 * kv_dim + self.d_model

        mlp_params = 3 * self.d_model * self.d_ff
        if self.bias:
            mlp_params += 2 * self.d_ff + self.d_model

        norm_params = 2 * self.d_model  # attn_norm + ffn_norm per block
        block_params = attn_params + mlp_params + norm_params

        final_norm_params = self.d_model
        return emb_params + lm_head_params + self.n_layers * block_params + final_norm_params

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def sciminilm_77m(
        cls,
        attention_variant: AttentionType = "mha",
        n_kv_heads: Optional[int] = None,
        **overrides,
    ) -> "ModelConfig":
        """77M-parameter preset (~77.02M params with MHA and tied embeddings).

        Architecture:
            vocab_size=32000, d_model=640, n_layers=12, n_heads=10,
            d_ff=1600, max_seq_len=2048, tie_word_embeddings=True.
        """
        if n_kv_heads is None:
            if attention_variant == "mha":
                n_kv_heads = 10
            elif attention_variant == "gqa":
                n_kv_heads = 2
            elif attention_variant == "mqa":
                n_kv_heads = 1
            else:
                raise ValueError(f"Unknown attention_variant: {attention_variant}")

        params = dict(
            vocab_size=32000,
            d_model=640,
            n_layers=12,
            n_heads=10,
            n_kv_heads=n_kv_heads,
            d_ff=1600,
            max_seq_len=2048,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
            dropout=0.0,
            tie_word_embeddings=True,
            use_flash_attention=True,
            use_triton_rmsnorm=True,
        )
        params.update(overrides)
        return cls(**params)

    @classmethod
    def tiny_test(
        cls,
        attention_variant: AttentionType = "gqa",
        n_kv_heads: Optional[int] = None,
        **overrides,
    ) -> "ModelConfig":
        """Tiny preset for fast CPU/GPU unit tests."""
        if n_kv_heads is None:
            if attention_variant == "mha":
                n_kv_heads = 4
            elif attention_variant == "gqa":
                n_kv_heads = 2
            elif attention_variant == "mqa":
                n_kv_heads = 1
            else:
                raise ValueError(f"Unknown attention_variant: {attention_variant}")

        params = dict(
            vocab_size=256,
            d_model=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=n_kv_heads,
            d_ff=128,
            max_seq_len=128,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
            dropout=0.0,
            tie_word_embeddings=True,
            use_flash_attention=True,
            use_triton_rmsnorm=True,
        )
        params.update(overrides)
        return cls(**params)


@dataclass
class QuantConfig:
    """Configuration for post-training weight quantization."""

    bits: Literal[4, 8] = 4
    group_size: int = 64
    symmetric: bool = True
    skip_modules: tuple[str, ...] = ("lm_head",)
    quantize_embeddings: bool = False


@dataclass
class InferenceConfig:
    """Configuration for autoregressive / speculative decoding."""

    max_new_tokens: int = 128
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    do_sample: bool = True
    eos_token_id: Optional[int] = None
    speculative_gamma: int = 4


@dataclass
class TrainConfig:
    """Configuration for pretraining SciMiniLM on 500M scientific-text tokens."""

    total_tokens: int = 500_000_000
    seq_len: int = 2048
    micro_batch_size: int = 8
    grad_accum_steps: int = 16
    learning_rate: float = 6e-4
    min_learning_rate: float = 6e-5
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 500
    eval_interval: int = 500
    eval_steps: int = 20
    save_interval: int = 1000
    log_interval: int = 10
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    checkpoint_dir: str = "checkpoints/sciminilm_77m"
    seed: int = 42
    num_workers: int = 0
    extra_metadata: dict = field(default_factory=dict)

    @property
    def tokens_per_step(self) -> int:
        return self.micro_batch_size * self.grad_accum_steps * self.seq_len

    @property
    def max_steps(self) -> int:
        return max(1, (self.total_tokens + self.tokens_per_step - 1) // self.tokens_per_step)
