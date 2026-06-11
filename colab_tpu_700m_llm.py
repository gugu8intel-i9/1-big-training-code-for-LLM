# One-file Google Colab TPU script: 700M-class experimental sparse MoE LLM
# Paste this whole file into a Colab code cell, or upload/run it with: !python colab_tpu_700m_llm.py
#
# IMPORTANT REALITY CHECK:
# - This builds a real JAX/Flax TPU training pipeline for a ~700M-parameter *experimental* model.
# - A free Colab TPU can only do toy/continued-pretraining scale. Broad, high-quality world knowledge requires
#   billions/trillions of tokens and far more compute than Colab. This script is designed to be scalable and honest.
# - The sparse-attention mask below is implemented as a research/prototype mask over dense score computation for
#   correctness/portability. Custom kernels are required for true long-context speedups.

import os, sys, math, time, json, random, subprocess, pathlib, itertools, functools, tempfile
from dataclasses import dataclass, asdict

# -----------------------------
# 0) Install dependencies in Colab
# -----------------------------
def pip_install():
    pkgs = [
        "-q", "flax>=0.8.4", "optax>=0.2.3", "orbax-checkpoint>=0.5.20",
        "datasets>=2.19.0", "sentencepiece>=0.2.0", "tqdm", "numpy"
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", *pkgs])

try:
    import google.colab  # noqa
    IN_COLAB = True
except Exception:
    IN_COLAB = False

if IN_COLAB and os.environ.get("SKIP_PIP_INSTALL", "0") != "1":
    pip_install()

# -----------------------------
# 1) Imports after install
# -----------------------------
import numpy as np
from tqdm.auto import tqdm
import sentencepiece as spm
from datasets import load_dataset, interleave_datasets

import jax
import jax.numpy as jnp
from jax import random as jr
import flax.linen as nn
from flax.training import train_state, common_utils
from flax import jax_utils
import optax
import orbax.checkpoint as ocp

print("JAX devices:", jax.devices())
print("Backend:", jax.default_backend())

# -----------------------------
# 2) Config
# -----------------------------
@dataclass
class Config:
    # Model: 700M-class total params with recursive weight tying.
    vocab_size: int = 32768
    d_model: int = 1536
    n_unique_layers: int = 7             # unique blocks; recursively reused
    recursions_per_layer: int = 4        # effective depth = 28
    n_heads: int = 16
    n_kv_heads: int = 4                  # grouped-query attention
    block_size: int = 512                # keep Colab-friendly; raise on bigger TPU
    attn_block: int = 64
    local_blocks: int = 3
    global_blocks: int = 1
    top_blocks: int = 4                  # MiniMax-style top-k block selection
    dropout: float = 0.0
    qk_norm: bool = True
    rope_theta: float = 10000.0

    # Fine-grained MoE: many small experts, top-k active.
    n_experts: int = 32
    top_k_experts: int = 4
    expert_hidden: int = 640             # 32 small experts ~= one coarse MoE split into fine experts
    router_jitter: float = 0.01

    # Multi-token prediction auxiliary heads.
    mtp_depth: int = 2
    mtp_weight: float = 0.15

    # Training
    seed: int = 42
    steps: int = 2000                    # increase for real runs
    warmup_steps: int = 200
    learning_rate: float = 2.0e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    label_smoothing: float = 0.001
    z_loss_weight: float = 1e-4
    load_balance_weight: float = 0.01
    router_z_loss_weight: float = 1e-3

    # My original optimization trick:
    # REAMS = Router Entropy-Annealed Markov Stabilization.
    # It starts with a mild adjacent-token router smoothness prior, then fades out so experts specialize later.
    reams_weight: float = 0.002
    reams_decay_steps: int = 1000

    # Batch; for Colab TPU v2/v3 start tiny, then tune upward.
    per_device_batch: int = 1
    grad_accum_steps: int = 1

    # Data/tokenizer/checkpoints
    workdir: str = "/content/markov_msa_tot_recursive_moe_700m" if IN_COLAB else "./runs/markov_msa_tot_recursive_moe_700m"
    tokenizer_prefix: str = "spm_32k"
    tokenizer_train_chars: int = 20_000_000
    save_every: int = 250
    eval_every: int = 100
    sample_every: int = 200

CFG = Config()
os.makedirs(CFG.workdir, exist_ok=True)
print(json.dumps(asdict(CFG), indent=2))

# -----------------------------
# 3) High-quality data mixture
# -----------------------------
# Public streaming datasets. Some may be temporarily unavailable; the loader skips failures.
DATA_MIX = [
    # English + broad world knowledge
    dict(path="HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", text="text", weight=0.40),
    dict(path="wikimedia/wikipedia", name="20231101.en", split="train", text="text", weight=0.20),
    # Math / reasoning
    dict(path="open-web-math/open-web-math", name=None, split="train", text="text", weight=0.20),
    dict(path="TIGER-Lab/MathInstruct", name=None, split="train", text="output", weight=0.10),
    dict(path="gsm8k", name="main", split="train", text=None, weight=0.05),
    # English books-style fallback-ish
    dict(path="roneneldan/TinyStories", name=None, split="train", text="text", weight=0.05),
]

FALLBACK_TEXTS = [
    "Mathematics is the study of patterns, quantity, structure, and logical proof.\n",
    "A good explanation uses definitions, examples, counterexamples, and clear reasoning.\n",
    "The Earth orbits the Sun. Water freezes near zero degrees Celsius at standard pressure.\n",
    "To solve a word problem, identify the variables, write equations, solve, and check units.\n",
]

def normalize_example(ex, spec):
    if spec["path"] == "gsm8k":
        return f"Question: {ex.get('question','')}\nThought: {ex.get('answer','')}\n"
    key = spec.get("text")
    if key is None:
        for k in ("text", "content", "output", "answer"):
            if k in ex:
                return str(ex[k])
        return ""
    return str(ex.get(key, ""))

def make_streaming_text_iterator():
    streams, probs = [], []
    for spec in DATA_MIX:
        try:
            kwargs = dict(path=spec["path"], split=spec["split"], streaming=True)
            if spec.get("name"):
                kwargs["name"] = spec["name"]
            ds = load_dataset(**kwargs)
            ds = ds.shuffle(buffer_size=10_000, seed=CFG.seed)
            ds = ds.map(lambda ex, spec=spec: {"_txt": normalize_example(ex, spec)})
            streams.append(ds)
            probs.append(float(spec["weight"]))
            print("Loaded", spec["path"])
        except Exception as e:
            print("Skipping dataset", spec["path"], "because", repr(e))
    if streams:
        probs = np.array(probs, dtype=np.float64); probs = probs / probs.sum()
        mixed = interleave_datasets(streams, probabilities=probs.tolist(), seed=CFG.seed, stopping_strategy="all_exhausted")
        for ex in mixed:
            txt = ex.get("_txt", "")
            if txt and len(txt) > 20:
                yield txt.replace("\x00", " ")
    else:
        while True:
            yield random.choice(FALLBACK_TEXTS)

# -----------------------------
# 4) Train/load SentencePiece tokenizer
# -----------------------------
def train_or_load_tokenizer():
    prefix = os.path.join(CFG.workdir, CFG.tokenizer_prefix)
    model_file = prefix + ".model"
    if os.path.exists(model_file):
        sp = spm.SentencePieceProcessor(model_file=model_file)
        print("Loaded tokenizer", model_file)
        return sp

    txt_path = os.path.join(CFG.workdir, "tokenizer_corpus.txt")
    print("Building tokenizer corpus...")
    n_chars = 0
    with open(txt_path, "w", encoding="utf-8") as f:
        for txt in make_streaming_text_iterator():
            # Include ToT/control tokens naturally in math-style examples.
            txt = txt.strip()
            if not txt: continue
            f.write(txt + "\n")
            n_chars += len(txt)
            if n_chars >= CFG.tokenizer_train_chars:
                break
    print("Training SentencePiece...")
    spm.SentencePieceTrainer.train(
        input=txt_path,
        model_prefix=prefix,
        vocab_size=CFG.vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        byte_fallback=True,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        user_defined_symbols="<thought>,</thought>,<branch>,</branch>,<score>,<final>"
    )
    return spm.SentencePieceProcessor(model_file=model_file)

SP = train_or_load_tokenizer()
BOS, EOS, PAD = SP.bos_id(), SP.eos_id(), SP.pad_id()

# -----------------------------
# 5) Packed token batch generator
# -----------------------------
def token_stream():
    for txt in make_streaming_text_iterator():
        # Add occasional Tree-of-Thought markup prompt style for reasoning data.
        if random.random() < 0.05:
            txt = "<thought> Consider multiple approaches. </thought>\n" + txt
        ids = [BOS] + SP.encode(txt, out_type=int) + [EOS]
        for i in ids:
            if 0 <= i < CFG.vocab_size:
                yield i

def batch_iterator():
    ndev = jax.local_device_count()
    bsz = ndev * CFG.per_device_batch
    length = CFG.block_size + CFG.mtp_depth + 1
    stream = token_stream()
    buf = []
    while True:
        while len(buf) < bsz * length:
            buf.append(next(stream))
        arr = np.array(buf[:bsz * length], dtype=np.int32).reshape(ndev, CFG.per_device_batch, length)
        del buf[:bsz * length]
        yield arr

# -----------------------------
# 6) Model components
# -----------------------------
def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    y = jnp.stack([-x2, x1], axis=-1)
    return y.reshape(x.shape)

def apply_rope(q, k, theta=10000.0):
    # q/k: [B,T,H,D]
    t = q.shape[1]
    d = q.shape[-1]
    freqs = 1.0 / (theta ** (jnp.arange(0, d, 2, dtype=jnp.float32) / d))
    pos = jnp.arange(t, dtype=jnp.float32)
    ang = pos[:, None] * freqs[None, :]
    sin = jnp.repeat(jnp.sin(ang), 2, axis=-1)[None, :, None, :]
    cos = jnp.repeat(jnp.cos(ang), 2, axis=-1)[None, :, None, :]
    return (q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin)

class RMSNorm(nn.Module):
    eps: float = 1e-6
    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.ones, (x.shape[-1],))
        x32 = x.astype(jnp.float32)
        return (x32 * jax.lax.rsqrt(jnp.mean(x32*x32, axis=-1, keepdims=True) + self.eps) * scale).astype(x.dtype)

class MinimaxSparseAttention(nn.Module):
    cfg: Config
    @nn.compact
    def __call__(self, x, train: bool):
        cfg = self.cfg
        B, T, D = x.shape
        H, KVH = cfg.n_heads, cfg.n_kv_heads
        HD = D // H
        q = nn.Dense(D, use_bias=False, dtype=jnp.bfloat16, name="q")(x).reshape(B,T,H,HD)
        k = nn.Dense(KVH*HD, use_bias=False, dtype=jnp.bfloat16, name="k")(x).reshape(B,T,KVH,HD)
        v = nn.Dense(KVH*HD, use_bias=False, dtype=jnp.bfloat16, name="v")(x).reshape(B,T,KVH,HD)
        if cfg.qk_norm:
            q = q / jnp.sqrt(jnp.maximum(jnp.sum(q.astype(jnp.float32)**2, axis=-1, keepdims=True), 1e-6)).astype(q.dtype)
            k = k / jnp.sqrt(jnp.maximum(jnp.sum(k.astype(jnp.float32)**2, axis=-1, keepdims=True), 1e-6)).astype(k.dtype)
        q, k = apply_rope(q, k, cfg.rope_theta)
        rep = H // KVH
        k = jnp.repeat(k, rep, axis=2)
        v = jnp.repeat(v, rep, axis=2)

        scores = jnp.einsum("bthd,bshd->bhts", q, k).astype(jnp.float32) / math.sqrt(HD)

        # MiniMax Sparse Attention approximation: robust min-max normalized block selector.
        assert T % cfg.attn_block == 0, "block_size must be divisible by attn_block"
        NB = T // cfg.attn_block
        qblk = q.reshape(B, NB, cfg.attn_block, H, HD).mean(axis=(2,3)).astype(jnp.float32)
        kblk = k.reshape(B, NB, cfg.attn_block, H, HD).mean(axis=(2,3)).astype(jnp.float32)
        bs = jnp.einsum("bnd,bmd->bnm", qblk, kblk) / math.sqrt(HD)
        # Causal block masking.
        bi = jnp.arange(NB)[:, None]
        bj = jnp.arange(NB)[None, :]
        causal_blk = bj <= bi
        bs = jnp.where(causal_blk[None, :, :], bs, -1e9)
        # Minimax normalization makes top-k less sensitive to outlier blocks.
        finite = bs > -1e8
        mn = jnp.min(jnp.where(finite, bs, 1e9), axis=-1, keepdims=True)
        mx = jnp.max(jnp.where(finite, bs, -1e9), axis=-1, keepdims=True)
        bs_norm = (bs - mn) / (mx - mn + 1e-6)
        _, idx = jax.lax.top_k(bs_norm, k=min(cfg.top_blocks, NB))
        selected = jnp.any(idx[..., None] == jnp.arange(NB)[None, None, None, :], axis=-2)
        local = (bj <= bi) & (bj >= (bi - cfg.local_blocks + 1))
        glob = (bj < cfg.global_blocks) & (bj <= bi)
        block_allowed = selected | local[None,:,:] | glob[None,:,:]

        tok_block = jnp.arange(T) // cfg.attn_block
        block_mask = block_allowed[:, tok_block[:,None], tok_block[None,:]]  # [B,T,T]
        causal_tok = jnp.arange(T)[None, :] <= jnp.arange(T)[:, None]
        mask = block_mask & causal_tok[None,:,:]
        scores = jnp.where(mask[:, None, :, :], scores, -1e9)
        att = nn.softmax(scores, axis=-1).astype(x.dtype)
        y = jnp.einsum("bhts,bshd->bthd", att, v).reshape(B,T,D)
        y = nn.Dense(D, use_bias=False, dtype=jnp.bfloat16, name="o")(y)
        return y

class FineGrainedMoE(nn.Module):
    cfg: Config
    @nn.compact
    def __call__(self, x, train: bool):
        cfg = self.cfg
        B,T,D = x.shape
        router_logits = nn.Dense(cfg.n_experts, use_bias=False, dtype=jnp.float32, name="router")(x.astype(jnp.float32))
        if train and cfg.router_jitter > 0:
            noise = jr.uniform(self.make_rng("dropout"), router_logits.shape, minval=1-cfg.router_jitter, maxval=1+cfg.router_jitter)
            router_logits = router_logits * noise
        router_probs = nn.softmax(router_logits, axis=-1)
        topv, topi = jax.lax.top_k(router_probs, cfg.top_k_experts)
        topv = topv / (jnp.sum(topv, axis=-1, keepdims=True) + 1e-9)
        dispatch = jnp.sum(nn.one_hot(topi, cfg.n_experts) * topv[..., None], axis=-2).astype(x.dtype)  # [B,T,E]

        # Vectorized small experts: SwiGLU FFN per expert.
        E,HID = cfg.n_experts, cfg.expert_hidden
        w1 = self.param("w1", nn.initializers.variance_scaling(1.0, "fan_in", "truncated_normal"), (E,D,HID), jnp.bfloat16)
        w3 = self.param("w3", nn.initializers.variance_scaling(1.0, "fan_in", "truncated_normal"), (E,D,HID), jnp.bfloat16)
        w2 = self.param("w2", nn.initializers.variance_scaling(1.0, "fan_in", "truncated_normal"), (E,HID,D), jnp.bfloat16)
        xbf = x.astype(jnp.bfloat16)
        a = jnp.einsum("btd,edh->bteh", xbf, w1)
        b = jnp.einsum("btd,edh->bteh", xbf, w3)
        h = nn.silu(a) * b
        y = jnp.einsum("bteh,ehd,bte->btd", h, w2, dispatch)
        aux = dict(router_probs=router_probs, router_logits=router_logits, top_indices=topi)
        return y.astype(x.dtype), aux

class TransformerBlock(nn.Module):
    cfg: Config
    @nn.compact
    def __call__(self, x, train: bool):
        cfg = self.cfg
        # Attention
        a = MinimaxSparseAttention(cfg, name="minimax_sparse_attn")(RMSNorm(name="attn_norm")(x), train)
        x = x + a
        # Markovian RSA = Markovian Recurrent State Augmentation: a learned one-step residual transition.
        prev = jnp.pad(x[:, :-1, :], ((0,0),(1,0),(0,0)))
        trans = nn.Dense(cfg.d_model, use_bias=False, dtype=jnp.bfloat16, name="markov_rsa_transition")(RMSNorm(name="markov_norm")(prev))
        gate = self.param("markov_rsa_gate", nn.initializers.constant(-2.0), (cfg.d_model,))
        x = x + nn.sigmoid(gate).astype(x.dtype) * trans
        # MoE FFN
        m, aux = FineGrainedMoE(cfg, name="fine_grained_moe")(RMSNorm(name="moe_norm")(x), train)
        x = x + m
        return x, aux

RematBlock = nn.remat(TransformerBlock, prevent_cse=False, static_argnums=(2,))

class RecursiveMoELM(nn.Module):
    cfg: Config
    @nn.compact
    def __call__(self, tokens, train: bool):
        cfg = self.cfg
        x = nn.Embed(cfg.vocab_size, cfg.d_model, dtype=jnp.bfloat16, name="tok_embed")(tokens)
        auxes = []
        for i in range(cfg.n_unique_layers):
            block = RematBlock(cfg, name=f"recursive_block_{i}")
            for r in range(cfg.recursions_per_layer):
                x, aux = block(x, train)
                auxes.append(aux)
        x = RMSNorm(name="final_norm")(x)
        emb = self.variables["params"]["tok_embed"]["embedding"]
        logits = jnp.einsum("btd,vd->btv", x.astype(jnp.float32), emb.astype(jnp.float32))
        # Multi-token prediction heads. These are cheap auxiliary heads, not used at inference.
        mtp_logits = []
        for k in range(cfg.mtp_depth):
            h = nn.Dense(cfg.d_model, dtype=jnp.bfloat16, name=f"mtp_proj_{k+1}")(x)
            mtp_logits.append(jnp.einsum("btd,vd->btv", h.astype(jnp.float32), emb.astype(jnp.float32)))
        return logits, mtp_logits, auxes

# -----------------------------
# 7) Loss, optimizer, pmap training
# -----------------------------
def count_params(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))

def cross_entropy_loss(logits, labels, smoothing=0.0):
    vocab = logits.shape[-1]
    logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    nll = -jnp.take_along_axis(logp, labels[...,None], axis=-1)[...,0]
    if smoothing > 0:
        smooth = -jnp.mean(logp, axis=-1)
        nll = (1-smoothing)*nll + smoothing*smooth
    return jnp.mean(nll)

def z_loss(logits):
    return jnp.mean(jax.nn.logsumexp(logits.astype(jnp.float32), axis=-1) ** 2)

class TrainState(train_state.TrainState):
    rng: jax.Array


def lr_schedule(step):
    warm = jnp.minimum(1.0, step / max(1, CFG.warmup_steps))
    progress = jnp.minimum(1.0, jnp.maximum(0.0, (step - CFG.warmup_steps) / max(1, CFG.steps - CFG.warmup_steps)))
    cosine = CFG.min_lr_ratio + 0.5*(1-CFG.min_lr_ratio)*(1+jnp.cos(jnp.pi*progress))
    return CFG.learning_rate * warm * cosine

def create_state(rng):
    model = RecursiveMoELM(CFG)
    dummy = jnp.zeros((CFG.per_device_batch, CFG.block_size), dtype=jnp.int32)
    variables = model.init({"params": rng, "dropout": rng}, dummy, True)
    params = variables["params"]
    print("Parameter count:", f"{count_params(params)/1e6:.1f}M")
    tx = optax.chain(
        optax.clip_by_global_norm(CFG.grad_clip),
        optax.adamw(learning_rate=lr_schedule, b1=0.9, b2=0.95, eps=1e-8, weight_decay=CFG.weight_decay)
    )
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx, rng=rng)

def loss_fn(params, state, batch):
    # batch [per_device_batch, block + mtp + 1]
    tokens = batch[:, :CFG.block_size]
    labels = batch[:, 1:CFG.block_size+1]
    rng, drng = jr.split(state.rng)
    logits, mtp_logits, auxes = state.apply_fn({"params": params}, tokens, True, rngs={"dropout": drng})
    loss = cross_entropy_loss(logits, labels, CFG.label_smoothing)
    metrics = {"xent": loss, "z": z_loss(logits)}
    loss = loss + CFG.z_loss_weight * metrics["z"]

    # Multi-token prediction for k-step future tokens.
    mtp_total = 0.0
    for k, ml in enumerate(mtp_logits, start=1):
        lab = batch[:, 1+k:CFG.block_size+1+k]
        mtp_total = mtp_total + cross_entropy_loss(ml, lab, CFG.label_smoothing)
    mtp_total = mtp_total / max(1, CFG.mtp_depth)
    loss = loss + CFG.mtp_weight * mtp_total
    metrics["mtp"] = mtp_total

    # MoE load-balance/router losses + REAMS.
    lb = 0.0; rz = 0.0; reams = 0.0
    for aux in auxes:
        p = aux["router_probs"].astype(jnp.float32)
        logits_r = aux["router_logits"].astype(jnp.float32)
        mean_p = jnp.mean(p, axis=(0,1))
        # Approx usage by top-k mask.
        topmask = jnp.max(nn.one_hot(aux["top_indices"], CFG.n_experts), axis=-2)
        mean_u = jnp.mean(topmask.astype(jnp.float32), axis=(0,1))
        lb = lb + CFG.n_experts * jnp.sum(mean_p * mean_u)
        rz = rz + jnp.mean(jax.nn.logsumexp(logits_r, axis=-1) ** 2)
        # REAMS: adjacent-token Markov router stabilization, annealed away over training.
        reams = reams + jnp.mean((p[:,1:,:] - jax.lax.stop_gradient(p[:,:-1,:]))**2)
    denom = max(1, len(auxes))
    lb, rz, reams = lb/denom, rz/denom, reams/denom
    anneal = jnp.maximum(0.0, 1.0 - state.step / max(1, CFG.reams_decay_steps))
    loss = loss + CFG.load_balance_weight*lb + CFG.router_z_loss_weight*rz + CFG.reams_weight*anneal*reams
    metrics.update({"loss": loss, "lb": lb, "router_z": rz, "reams": reams, "lr": lr_schedule(state.step)})
    return loss, metrics

@functools.partial(jax.pmap, axis_name="data")
def train_step(state, batch):
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params, state, batch)
    grads = jax.lax.pmean(grads, axis_name="data")
    metrics = jax.lax.pmean(metrics, axis_name="data")
    new_state = state.apply_gradients(grads=grads)
    new_state = new_state.replace(rng=jr.split(state.rng)[0])
    return new_state, metrics

@functools.partial(jax.pmap, axis_name="data")
def eval_step(state, batch):
    _, metrics = loss_fn(state.params, state, batch)
    return jax.lax.pmean(metrics, axis_name="data")

# -----------------------------
# 8) Sampling + Tree of Thoughts inference
# -----------------------------
def softmax_np(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def generate_one(state_unrep, prompt, max_new=128, temp=0.8, top_p=0.95, seed=0):
    model = RecursiveMoELM(CFG)
    ids = [BOS] + SP.encode(prompt, out_type=int)
    rng = np.random.default_rng(seed)
    for _ in range(max_new):
        ctx = ids[-CFG.block_size:]
        arr = np.full((1, CFG.block_size), PAD, dtype=np.int32)
        arr[0, -len(ctx):] = ctx
        logits, _, _ = model.apply({"params": state_unrep.params}, jnp.array(arr), False)
        l = np.array(logits[0, -1]) / max(temp, 1e-4)
        probs = softmax_np(l)
        order = np.argsort(probs)[::-1]
        cdf = np.cumsum(probs[order])
        keep = order[cdf <= top_p]
        if len(keep) == 0: keep = order[:1]
        p = probs[keep]; p = p / p.sum()
        nxt = int(rng.choice(keep, p=p))
        ids.append(nxt)
        if nxt == EOS: break
    return SP.decode(ids[1:])

def tree_of_thoughts_generate(state_unrep, question, branches=3, depth=3, max_thought_tokens=80):
    """Simple ToT inference wrapper: branch, self-score by average logprob proxy, keep best."""
    frontier = [f"Question: {question}\n"]
    for d in range(depth):
        candidates = []
        for base in frontier:
            for b in range(branches):
                thought = generate_one(state_unrep, base + f"<branch> {b}\n<thought>", max_new=max_thought_tokens, temp=0.9, seed=1000+d*31+b)
                candidates.append(base + "<thought>" + thought.split("<thought>")[-1] + "</thought>\n")
        # Lightweight heuristic scoring: prefer concise candidates containing math/reasoning words.
        def score(s):
            keys = ["therefore", "because", "=", "answer", "so", "let"]
            return sum(k in s.lower() for k in keys) - 0.001*len(s)
        candidates.sort(key=score, reverse=True)
        frontier = candidates[:branches]
    return generate_one(state_unrep, frontier[0] + "<final>", max_new=128, temp=0.7, seed=999)

# -----------------------------
# 9) Checkpointing
# -----------------------------
ckpt_dir = os.path.join(CFG.workdir, "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)
checkpointer = ocp.PyTreeCheckpointer()

def save_ckpt(state, step):
    unrep = jax_utils.unreplicate(state)
    path = os.path.join(ckpt_dir, f"step_{int(step)}")
    checkpointer.save(path, unrep, force=True)
    with open(os.path.join(CFG.workdir, "config.json"), "w") as f:
        json.dump(asdict(CFG), f, indent=2)
    print("Saved", path)

# -----------------------------
# 10) Train
# -----------------------------
def main():
    rng = jr.PRNGKey(CFG.seed)
    state = create_state(rng)
    state = jax_utils.replicate(state)
    it = batch_iterator()
    t0 = time.time()
    for step in range(1, CFG.steps + 1):
        batch = next(it)
        state, metrics = train_step(state, batch)
        if step % 10 == 0:
            m = jax.tree_util.tree_map(lambda x: float(np.array(x)[0]), metrics)
            toks = step * jax.local_device_count() * CFG.per_device_batch * CFG.block_size
            print(f"step {step:6d} loss {m['loss']:.4f} xent {m['xent']:.4f} mtp {m['mtp']:.4f} lb {m['lb']:.3f} reams {m['reams']:.5f} lr {m['lr']:.2e} tok/s {toks/(time.time()-t0):.1f}")
        if step % CFG.eval_every == 0:
            eb = next(it)
            em = eval_step(state, eb)
            em = jax.tree_util.tree_map(lambda x: float(np.array(x)[0]), em)
            print("eval", em)
        if step % CFG.sample_every == 0:
            unrep = jax_utils.unreplicate(state)
            print(generate_one(unrep, "Question: If a train travels 60 km in 2 hours, what is its speed?\nThought:", max_new=80, seed=step))
        if step % CFG.save_every == 0:
            save_ckpt(state, step)
    save_ckpt(state, CFG.steps)
    print("Try ToT after training:")
    print(tree_of_thoughts_generate(jax_utils.unreplicate(state), "A rectangle has area 48 and width 6. What is its length?"))

if __name__ == "__main__":
    main()
