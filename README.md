# One Big Training Code for a 700M-Class Experimental LLM on Colab TPU

This repository contains a single pasteable Google Colab script:

- [`colab_tpu_700m_llm.py`](./colab_tpu_700m_llm.py)

It builds and trains a **700M-class experimental decoder-only language model** using JAX/Flax on a Colab TPU. The code is intentionally honest about Colab limits: it can instantiate the architecture and run toy or small continued-pretraining jobs, but a broadly knowledgeable 700M model requires far more tokens and compute than free Colab provides.

## What the script implements

- **Markovian RSA**: implemented as **Markovian Recurrent State Augmentation**, a learned one-step residual transition from the previous token state into the current token state.
- **MiniMax-style Sparse Attention**: block-sparse causal masking with local blocks, global sink blocks, and top-k content-selected KV blocks using min-max-normalized block scores.
- **Tree of Thoughts (ToT)**: control tokens are added to the tokenizer, math/reasoning samples are formatted with thought markers where possible, and a simple ToT branch/search inference wrapper is included.
- **Recursive layers**: 7 unique transformer blocks are recursively reused 4 times each, producing 28 effective depth steps while reducing unique parameter count versus 28 separate blocks.
- **Fine-grained MoE**: many small SwiGLU experts with top-k routing per token, inspired by recent fine-grained MoE designs.
- **Multi-token prediction**: auxiliary future-token heads for better representation learning and possible speculative-decoding compatibility.
- **TPU-oriented training tricks**: bf16 parameters/compute where appropriate, JAX `pmap`, rematerialization/gradient checkpointing, GQA, RoPE, RMSNorm, AdamW, warmup+cosine schedule, gradient clipping, label smoothing, z-loss, router z-loss, load balancing, and streaming datasets.
- **Original optimization trick**: **REAMS** — Router Entropy-Annealed Markov Stabilization. Early in training, adjacent-token router distributions are mildly stabilized; the regularizer anneals away so experts can specialize.

## Dataset mixture

The script attempts to stream public datasets for:

- English and world knowledge: FineWeb-Edu, Wikipedia
- Math/reasoning: OpenWebMath, MathInstruct, GSM8K
- English fallback: TinyStories

If a dataset is unavailable in Colab, it is skipped automatically. If all fail, a tiny fallback text stream is used so the script still runs.

## Usage in Google Colab

1. Open a Colab notebook.
2. Runtime → Change runtime type → select **TPU**.
3. Paste the entire contents of `colab_tpu_700m_llm.py` into a cell.
4. Run it.

For a faster smoke test, edit near the top:

```python
CFG.steps = 10
CFG.tokenizer_train_chars = 200_000
CFG.block_size = 256
CFG.attn_block = 64
```

For a larger run, increase `steps`, tokenizer corpus size, sequence length, and batch size only if the TPU has enough memory.

## Important limitation

The script creates a real 700M-class model, but **training a high-quality broad-knowledge LLM from scratch is not possible on a single free Colab TPU**. You would normally need:

- hundreds of billions to trillions of tokens,
- many TPU/GPU days or weeks,
- robust distributed checkpointing,
- deduplicated/filtered data pipelines,
- extensive evaluation and safety testing.

This repo is best viewed as a compact research/education scaffold for experimenting with the requested architectural ingredients.
