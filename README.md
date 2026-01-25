# Domain-Specific Fine-Tuning for Software Translation

Fine-tuning encoder-decoder and decoder-only models (mBART/NLLB, Llama/Mistral with QLoRA) on domain-specific parallel data (English ↔ Dutch software/UI strings).

Project structure and initial setup.

EN→NL Software-Domain Translation Fine-Tuning (Part A + Part B)

This repo implements a **domain-specific fine-tuning pipeline** for English → Dutch translation.

It contains:
- **Part A:** Encoder–Decoder (Seq2Seq) fine-tuning using a translation model (MarianMT).
- **Part B:** Decoder-only (Causal LM) instruction tuning using **QLoRA** (4-bit + LoRA) for Colab-friendly training.

Evaluations:
1) **FLORES devtest** (general domain MT benchmark)
2) **Provided XLSX** software-domain test set (`Dataset_Challenge_1.xlsx`)

---

## Compute Constraints (Google Colab Free)
This project is designed to run on **Google Colab Free**, which imposes:
- limited GPU VRAM (varies by runtime)
- limited session time
- training budget constraints (hence lower epochs / dataset caps)

Therefore:
- **Part A** trains with 1–2 epochs for a fast iteration baseline
- **Part B** uses **QLoRA** (4-bit quantization + LoRA) to make decoder-only tuning feasible
- dataset rows are capped to keep runtime manageable

---

## Why these models?

### Part A (Seq2Seq)
Encoder–decoder models are architecturally aligned with translation tasks:
- encoder reads source
- decoder generates target

We use:
- `Helsinki-NLP/opus-mt-en-nl` (MarianMT EN→NL)

### Part B (Decoder-only)
Decoder-only LMs are not native MT models, so translation is taught via **instruction tuning**:
- Prompt: “Translate the following English text into Dutch…”
- Model learns to output only the Dutch translation

We use a modern instruct multilingual causal LM:
- `Qwen/Qwen2.5-0.5B-Instruct` (safer for Colab Free)
- Optionally `Qwen/Qwen2.5-1.5B-Instruct` (better quality if VRAM allows)

---

## Metrics: BLEU + chrF
We report:
- **BLEU (sacreBLEU):** standard MT n-gram overlap metric
- **chrF:** character F-score, more robust for morphology and minor tokenization variance

Note: BLEU can be harsh for decoder-only outputs (which may paraphrase / add punctuation), so chrF helps interpret translation quality.

---

## How to Run

### Install
```bash
pip install -r requirements.txt