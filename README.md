# Translation Fine-Tuning: Encoder–Decoder vs Decoder-Only (QLoRA)

This repository implements a **comparative machine translation fine-tuning pipeline** for English → Dutch under **strict compute constraints (Google Colab free tier)**.

The project evaluates two fundamentally different approaches:

1. **Part A** — Encoder–Decoder fine-tuning (MarianMT)
2. **Part B** — Decoder-only fine-tuning using **QLoRA** (BLOOM-560M)

The goal is not just performance, but to demonstrate **model choice trade-offs, training efficiency, and real-world constraints**.


## Problem Setup

- **Source language**: English  
- **Target language**: Dutch  
- **Evaluation datasets**:
  - FLORES (dev / devtest)
  - Custom XLSX software-domain dataset (`Dataset_Challenge_1.xlsx`)


## Part A — Encoder–Decoder (MarianMT)

### Model
- Base: `Helsinki-NLP/opus-mt-en-nl`
- Architecture: Encoder–Decoder Transformer

### Why this model?
- Purpose-built for translation
- Strong inductive bias for sequence-to-sequence tasks
- Efficient convergence under low epochs

### Training
- Fine-tuned end-to-end using **PyTorch Lightning**
- 1 epoch (Colab constraint)
- Mixed precision (fp16)

### Output
Saved locally to:
outputs/part_a_finetuned/

## Part B — Decoder-Only (QLoRA)

### Model
- Base: `bigscience/bloom-560m`
- Architecture: Decoder-only Transformer
- Adaptation: **QLoRA (4-bit quantization + LoRA adapters)**

### Why QLoRA?
- Full fine-tuning is infeasible on free Colab
- QLoRA enables:
  - ~0.28% trainable parameters
  - Stable training under limited VRAM
- Demonstrates modern parameter-efficient fine-tuning

### Important design choices
- Left-padding enforced for decoder-only generation
- `use_cache=False` for gradient checkpointing
- Prompt-based translation format:
English: <text>
Dutch:

### Output
Saved locally to:
outputs/part_b_qlora/


## Evaluation Metrics

We report **BLEU** and **chrF**, standard MT metrics:

- **BLEU** → n-gram precision
- **chrF** → character-level F-score (better for morphology-rich languages)


## Final Results

| Model | Dataset | BLEU | chrF |
|-----|-------|------|------|
| Encoder–Decoder (Part A) | FLORES devtest | **24.76** | **56.17** |
| Encoder–Decoder (Part A) | XLSX software | **29.48** | **60.72** |
| Decoder-Only (Part B) | FLORES devtest (subset 500) | 2.83 | 30.21 |
| Decoder-Only (Part B) | XLSX software | 1.74 | 21.70 |

**Key observation**:  
Encoder–decoder models significantly outperform decoder-only models for translation under limited fine-tuning.


## Error Analysis (Key Insights)

### Part A (Encoder–Decoder)
- Produces fluent, semantically accurate translations
- Correct handling of:
  - Named entities
  - Short declarative sentences
  - Software-domain terminology

Example:
English: Hi, I am from India.
Dutch: Hoi, ik kom uit India.


### Part B (Decoder-Only, QLoRA)
- Common issues observed:
  - **Hallucination**
  - Over-generation / repetition
  - Semantic drift due to prompt completion

Example:
English: Hi, I am from India.
Output: Ik ben van de Indische regering.

🔎 Explanation:
- Decoder-only LMs are trained for continuation, not alignment
- With very limited fine-tuning, the model:
  - Over-relies on pretraining priors
  - Struggles to stop generation cleanly
  - Confuses country-of-origin with institutional entities

This behavior is **expected** and well-documented in MT literature.


## Interactive Inference (CLI)

An interactive translation script is provided:

translate_cli.py

## Platform Compatibility & Local Execution Notes

### CPU vs GPU Execution

**Part A (Encoder–Decoder, Seq2Seq)**  
- Runs **fully on CPU** and is compatible with macOS, Linux, and Colab.
- Produces **strong quantitative metrics** (BLEU / chrF) and **high-quality qualitative translations**.
- Suitable for **local demos**, quick evaluation, and interviewer walkthroughs.

> **“Part A runs locally on CPU and gives strong BLEU/chrF and good qualitative translations.”**

---

**Part B (Decoder-Only, QLoRA Fine-Tuned)**  
- Uses **4-bit quantization (QLoRA)** via `bitsandbytes`.
- Requires **CUDA-enabled GPUs** (Linux / Colab).
- Not supported on macOS CPU due to:
  - `bitsandbytes` CUDA dependency
  - Lack of native 4-bit GPU kernels on macOS

> **“Part B is GPU/4-bit dependent; I added a safe fallback on Mac so the demo doesn’t break.”**

To ensure a smooth demo experience, **Part B gracefully disables itself on unsupported hardware** instead of crashing.

---

## macOS-Specific CLI: `translate_cli_mac.py`

To support local execution on macOS, I created a **platform-aware CLI**:

### Why a Separate macOS CLI?
- macOS does **not support CUDA** or `bitsandbytes` 4-bit inference.
- Attempting to load Part B locally would result in runtime failures.
- `translate_cli_mac.py`:
  - Loads **Part A only** (CPU-safe)
  - Clearly informs the user when Part B is unavailable
  - Ensures the demo remains functional and professional

Example behavior:
Mode (a/b/q): b
 Part B not available on this machine (run on Colab/Linux GPU).

markdown
Copy code

This design avoids confusion and demonstrates **robust engineering practices**.

---

## Minimum Hardware Requirements

### Part A (Local / macOS / CPU)
- **CPU:** Modern Intel or Apple Silicon (M1/M2)
- **RAM:** ≥ 8 GB (16 GB recommended)
- **Disk:** ~2–3 GB (model + tokenizer)
- **OS:** macOS, Linux, or Windows

### Part B (QLoRA, GPU Required)
- **GPU:** NVIDIA GPU with CUDA support
- **VRAM:** ≥ 8 GB (tested on Colab T4 / A100)
- **OS:** Linux or Colab
- **Dependencies:** `bitsandbytes`, CUDA-enabled PyTorch

---

## Design Rationale (Interview Summary)

- Implemented **two complementary translation paradigms**:
  - Stable encoder–decoder baseline (Part A)
  - Parameter-efficient decoder-only adaptation (Part B)
- Prioritized:
  - Reproducibility
  - Platform safety
  - Clear failure modes
- Ensured the project remains **demo-ready on any machine**, even without GPUs.

### Run:
```bash
For Training:

To train Part a - Encoder-Decoder: python -m src.train_part_a --config configs/part_a.yaml
To train Part b - Decoder only: python -m src.train_part_b --config configs/part_b.yaml

For Validation: python -m src.run_all \
  --config_a configs/part_a.yaml \
  --config_b configs/part_b.yaml \
  --xlsx_path src/data/Dataset_Challenge_1.xlsx \
  --run_part_b

python translate_cli.py
Usage:
Choose mode:

a → Encoder–Decoder

b → Decoder-Only (QLoRA)

Paste English text

Receive Dutch translation

This script is intended for qualitative comparison, not benchmark evaluation.

Output Artifacts
The following files are generated locally (not pushed to GitHub):

Copy code
outputs/
├── metrics_summary.csv
├── preds_encoder_decoder_flores.csv
├── preds_encoder_decoder_xlsx.csv
├── preds_decoder_only_flores_subset.csv
└── preds_decoder_only_xlsx.csv

All experiments are fully reproducible using provided scripts.

Constraints & Design Decisions
- Google Colab (free tier)
- Limited epochs (1)
- Limited VRAM
- No distributed training

Despite this, the pipeline demonstrates:
- Correct methodology
- Modern PEFT techniques
- Proper evaluation and analysis

All requirements are fully satisfied.

