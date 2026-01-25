# translate_cli.py
# Run in Colab terminal:
#   python translate_cli.py
# or:
#   python translate_cli.py --repo_dir /content/translation-finetune

import argparse
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

def load_part_a(repo_dir: str, rel_dir: str = "outputs/part_a_finetuned"):
    path = os.path.join(repo_dir, rel_dir)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Part A directory not found: {path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path).to(device)
    model.eval()
    return model, tok, device


def load_part_b(repo_dir: str, rel_dir: str = "outputs/part_b_qlora"):
    path = os.path.join(repo_dir, rel_dir)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Part B directory not found: {path}")

    tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    # IMPORTANT for decoder-only generation:
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # device_map="auto" will place on GPU if available; else CPU
    model = AutoModelForCausalLM.from_pretrained(path, device_map="auto")
    model.eval()
    return model, tok


@torch.no_grad()
def translate_part_a(model, tok, device, text: str, max_src_len=128, max_new_tokens=128, num_beams=4):
    enc = tok(
        [text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_src_len
    ).to(device)

    gen = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=False,
    )
    return tok.batch_decode(gen, skip_special_tokens=True)[0]


@torch.no_grad()
def translate_part_b(model, tok, text: str, max_len=256, max_new_tokens=128, num_beams=4):
    def build_prompt(src: str) -> str:
        return (
            "You are a professional software localization translator.\n"
            "Translate the following English text into fluent Dutch.\n"
            "Return ONLY the Dutch translation.\n\n"
            f"English: {src}\nDutch:"
        )

    prompt = build_prompt(text)
    enc = tok(
        [prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len
    )
    # device_map="auto" -> model.device works
    enc = {k: v.to(model.device) for k, v in enc.items()}

    gen = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=num_beams,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    decoded = tok.batch_decode(gen, skip_special_tokens=True)[0]
    # Take only what comes after the FIRST "Dutch:"
    out = decoded.split("Dutch:", 1)[1] if "Dutch:" in decoded else decoded

    # Stop if the model starts repeating templates again
    stop_markers = ["\nEnglish:", "\nDutch:", "\nHuman:", "\nAssistant:"]
    for m in stop_markers:
        if m in out:
            out = out.split(m, 1)[0]

    # Keep just the first non-empty line (usually the best translation)
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return lines[0] if lines else out.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_dir", default="/content/translation-finetune")
    ap.add_argument("--part_a_dir", default="outputs/part_a_finetuned")
    ap.add_argument("--part_b_dir", default="outputs/part_b_qlora")
    ap.add_argument("--max_src_len", type=int, default=128)
    ap.add_argument("--max_len", type=int, default=256)  # decoder-only input length
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--num_beams", type=int, default=4)
    args = ap.parse_args()

    repo_dir = os.path.abspath(args.repo_dir)
    if not os.path.isdir(repo_dir):
        print(f"ERROR: repo_dir not found: {repo_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Repo: {repo_dir}")
    print("CUDA available:", torch.cuda.is_available())
    print("\nLoading Part A (encoder-decoder)...")
    a_model, a_tok, a_device = load_part_a(repo_dir, args.part_a_dir)
    print("✅ Part A loaded.")

    print("\nLoading Part B (decoder-only QLoRA)...")
    b_model, b_tok = load_part_b(repo_dir, args.part_b_dir)
    print("✅ Part B loaded.")

    print("\n--- Interactive translator ---")
    print("Type mode: a | b  (or 'q' to quit)")
    print("Then paste English text. Press Enter.\n")

    while True:
        mode = input("Mode (a/b/q): ").strip().lower()
        if mode in {"q", "quit", "exit"}:
            break
        if mode not in {"a", "b"}:
            print("Please enter 'a' (Part A), 'b' (Part B), or 'q' to quit.")
            continue

        text = input("English: ").strip()
        if not text:
            print("Empty input. Try again.")
            continue

        if mode == "a":
            out = translate_part_a(
                a_model, a_tok, a_device,
                text,
                max_src_len=args.max_src_len,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams
            )
            print("Dutch (Part A):", out)
        else:
            out = translate_part_b(
                b_model, b_tok,
                text,
                max_len=args.max_len,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams
            )
            print("Dutch (Part B):", out)

        print("-" * 70)

    print("Bye!")


if __name__ == "__main__":
    main()
