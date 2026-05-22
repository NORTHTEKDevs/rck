# RCK Polisher Training Runbook

Concrete instructions for training the v7 distilled polisher -- the small
language model that turns RCK's template-rendered drafts into fluent
prose. This is the ONLY $-positive milestone in the v10 → parity plan.

## What you'll produce

A 5M-80M parameter PyTorch transformer saved as a checkpoint directory
containing:

  * `model.pt` -- weights
  * `tokenizer.json` -- vocab
  * `config.json` -- architecture spec

Load it via `rck.polisher.NeuralPolisher(weights_path)` and plug into
`InvertedLM(polisher=...)`. Replaces the v4 RuleBasedPolisher.

## What you'll need

  * **A GPU.** Minimum: 12GB VRAM (e.g. RTX 4060, 4070, 3090). Ideal:
    A100 (40GB) on RunPod / Lambda / Vast (~$1-2/hr).
  * **CUDA-enabled PyTorch.** The CPU torch we ship in `pyproject.toml`
    works for inference but is too slow for training.
    `pip install torch --index-url https://download.pytorch.org/whl/cu121`
  * **Disk:** ~5 GB for the corpus + checkpoints.
  * **Time:** see table below.

## Sizes + recommended run

| size | params | hidden | layers | heads | batch | LR | steps | epochs (on 56k corpus) | time / GPU |
|---|---|---|---|---|---|---|---|---|---|
| tiny   |   0.16M |  64 | 2 | 4 |  16 | 3e-3 |   500 | ~1   | 30s / CPU smoke |
| small  |   5M    | 128 | 4 | 4 |  32 | 3e-4 |  5000 | ~3   | 15 min / RTX 4060 |
| medium |  25M    | 256 | 6 | 8 |  64 | 3e-4 | 15000 | ~10  | 1 hr / RTX 4090 |
| large  |  80M    | 512 | 8 | 8 |  64 | 2e-4 | 30000 | ~30  | 3-4 hr / A100 |

`small` and `medium` produce good results at our v6 corpus scale (56k
pairs). `large` is recommended once you've bulk-imported ConceptNet
(~3M facts → ~10M training pairs).

## Step-by-step

### 1. Generate the training corpus

```bash
cd ~/projects/active/rck
python scripts/build_training_corpus.py \
    --out data/training_corpus.jsonl \
    --examples-per-triple 8 \
    --substitutions-per-pair 3
```

At v6 KB size this produces ~150k examples (~20 MB). Once ConceptNet
is imported expect ~3-5M examples.

### 2. Smoke test (CPU, 30 seconds)

Verify the pipeline before paying for GPU:

```bash
python scripts/smoke_train.py \
    --max-examples 2000 \
    --steps 200 \
    --batch-size 8
```

Expected: loss drops from ~7 to ~2 in <30 seconds. Output is gibberish
(tiny model, tiny dataset) but pipeline integrity confirmed.

### 3. Provision GPU

If renting:

```bash
# RunPod / Lambda / Vast.ai: pick PyTorch 2.x + CUDA 12.x AMI
# Recommended box: 1x A100 40GB at ~$1.20/hr
```

Otherwise jump to step 4 with your local GPU.

### 4. Install CUDA torch + clone repo

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
git clone <repo>; cd rck; pip install -e .
```

### 5. Run the real training

```bash
python scripts/train_polisher_real.py \
    --corpus data/training_corpus.jsonl \
    --out checkpoints/polisher_v7 \
    --size small \
    --steps 5000 \
    --batch-size 32 \
    --device cuda
```

Watch for: loss decreasing monotonically (modulo batch noise). Expect
final loss in the 0.3-0.7 range on small model, lower on bigger.

### 6. Validate

```bash
python -c "
from rck.polisher import NeuralPolisher
p = NeuralPolisher('checkpoints/polisher_v7', device='cuda')
for d in ['the dog is a mammal',
          'the capital of france is paris',
          'shakespeare wrote hamlet']:
    print(f'{d}  ->  {p.polish(d)}')"
```

If outputs look like fluent rephrasings, you have a working v7 polisher.

### 7. Plug into RCK

```python
from rck.conscious_agent import ConsciousAgent
from rck.inverted_lm import InvertedLM
from rck.polisher import NeuralPolisher

agent = ConsciousAgent(dim=4096, n_shards=128, seed=0)
# (load KBs as usual)
polisher = NeuralPolisher("checkpoints/polisher_v7", device="cuda")
inv = InvertedLM(agent=agent, polisher=polisher)
print(inv.generate("What color is the sky?")["response"])
```

## Expected costs

  * **Local with existing GPU:** $0 + electricity.
  * **RunPod A100 (small):** ~$0.20 (15 min × $1.20/hr).
  * **RunPod A100 (medium):** ~$1.30 (~1 hr).
  * **RunPod A100 (large):** ~$5 (~3-4 hr).
  * **+1B-token corpus (post-ConceptNet) at large:** ~$20-50.

The most expensive realistic run -- post-ConceptNet, large model, full
training -- is still <$100. For context, GPT-4 training was estimated at
$60-100M.

## Troubleshooting

  * **Loss not decreasing**: lower LR (try 1e-4 → 5e-5), increase
    warmup steps (try 500), verify your corpus isn't all the same pair.
  * **OOM**: reduce batch size, or reduce `max_seq_len` in
    `PolisherConfig`.
  * **Slow on CPU**: expected. CPU is for smoke testing only.
  * **`Cuda not available`**: install CUDA-enabled torch (see step 4).

## Going further

Once the v7 small/medium polisher is working:

  * **v7.1 dialogue-tuned**: fine-tune on conversational pairs.
  * **v7.2 creative head**: separate LM for open-ended generation.
  * **v7.3 code-aware**: fine-tune on (intent, code) pairs from open
    source.

All of these are additional ~$1-10 runs once you have the base v7
infrastructure working.

## The thesis being validated

If the v7 polisher produces prose that reads naturally, the
factor-the-problem architecture hypothesis is validated. RCK at v7 +
post-ConceptNet KB is a viable alternative to GPT-class systems for
the majority of practical use cases, at 30,000x less compute cost,
with structurally superior properties on auditability, editability,
continual learning, and latency.

If the polisher produces stilted output even at large scale, the
hypothesis fails at the boundary between "templates can produce all
the surface structure language needs" and reality. That tells us
something important: that scale really does discover composition
nobody hand-codes. We'd then need a different approach: maybe a much
larger LM head, or a different decomposition entirely.

Either way, the experiment is cheap (<$100) and the answer is
informative.
